"""Alien invasion mini-game for seed forging."""

from __future__ import annotations

import hashlib
import math
import random
import time
from pathlib import Path
from typing import Callable

import yaml

try:  # pragma: no cover - Tk only available with GUI rendering
    import tkinter as tkinter_module
    from tkinter import ttk as tkinter_ttk
except Exception:  # pragma: no cover - executed when Tk is unavailable
    tkinter_module = None
    tkinter_ttk = None

from rng_minigames.api import MinigameContext

from .ai_agent import AlienInvasionAI
from .ai_config import load_settings as load_ai_settings
from .game_config import load_settings as load_game_settings
from .hall_of_fame import HallOfFame


def launch_alien_invasion(context: MinigameContext) -> None:
    """Space-invader inspired mini-game."""

    def _apply_seed(order: list[str], duration: float) -> None:
        """Derive a deterministic seed from the selected enemy order."""
        payload = "|".join(order) + f"|{int(duration * 1000)}"
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        seed_value = str(int(digest[:14], 16))
        context.set_seed(seed_value)
        context.notify(
            f"Alien invasion forged seed {seed_value}.",
            "INFO",
        )

    if not context.render or tkinter_module is None or context.tk_root is None:
        dummy_order = [f"E{i}" for i in range(10)]
        random.shuffle(dummy_order)
        _apply_seed(dummy_order, random.random() * 20)
        return

    ai_settings = load_ai_settings()
    game_settings = load_game_settings()
    player_cfg = game_settings.get("player", {})
    general_cfg = game_settings.get("general", {})
    charges_cfg = game_settings.get("charges", {})
    motion_cfg = game_settings.get("player_motion", {})
    explosion_cfg = game_settings.get("explosion", {})
    player_explosion_cfg = game_settings.get("player_explosion", {})
    debris_cfg = game_settings.get("debris", {})
    default_learning_speed = int(ai_settings.get("learning_speed", 10))
    max_run_seconds = int(ai_settings.get("run_duration_seconds", 300))

    storage_dir = Path(__file__).resolve().parent / "_storage"
    storage_dir.mkdir(exist_ok=True)
    hall_of_fame = HallOfFame(storage_dir)
    ai_brain = AlienInvasionAI(storage_dir)
    ai_brain.begin_run()
    learning_history_path = storage_dir / "ai_learning_stats.yml"
    ALLTIME_LEARNING_DEFAULTS = {
        "runs": 0,
        "wins": 0,
        "losses": 0,
        "kill_total": 0.0,
        "kill_samples": 0,
        "edge_total": 0.0,
        "edge_samples": 0,
    }

    def _persist_learning_history() -> None:
        """Save the long-term learning statistics to disk."""
        try:
            learning_history_path.write_text(
                yaml.safe_dump(learning_alltime_stats, sort_keys=False)
            )
        except Exception:
            pass

    def _load_learning_history() -> dict[str, float | int]:
        """Load archived learning stats or reset to defaults."""
        if not learning_history_path.exists():
            return dict(ALLTIME_LEARNING_DEFAULTS)
        try:
            raw = yaml.safe_load(learning_history_path.read_text()) or {}
        except Exception:
            raw = {}
        stats: dict[str, float | int] = {}
        for key, default in ALLTIME_LEARNING_DEFAULTS.items():
            stat_value = raw.get(key, default)
            stats[key] = stat_value
        return stats

    def _reset_learning_history() -> None:
        """Reset the persisted learning stats to their defaults."""
        learning_alltime_stats.clear()
        learning_alltime_stats.update(dict(ALLTIME_LEARNING_DEFAULTS))
        _persist_learning_history()

    learning_alltime_stats = _load_learning_history()
    _persist_learning_history()

    window = tkinter_module.Toplevel(context.tk_root)
    window.title("Alien invasion")
    window.resizable(False, False)
    window.transient(context.tk_root)
    canvas_width = int(760 * 1.5)
    canvas_height = int(480 * 1.5)
    canvas = tkinter_module.Canvas(
        window,
        width=canvas_width,
        height=canvas_height,
        highlightthickness=0,
        background="#040912",
    )
    canvas.pack(padx=16, pady=(0, 8))
    field_margin = 80
    ground_height = 30
    sky_height = canvas_height - ground_height
    paused = False
    learning_mode = False
    learning_restart_handle: str | None = None
    shooting_stars: list[dict] = []
    next_shooting_star_time = time.time() + random.expovariate(0.1)
    learning_speed_multiplier = default_learning_speed
    learning_speed_min = 1
    learning_speed_max = 60
    base_tick_delay_ms = 40
    min_event_delay_ms = 8
    base_ticks_per_second = 1000 / base_tick_delay_ms
    learning_speed_var: "tkinter_module.IntVar | None" = None
    player_explosion: list[dict] = []
    player_explosion_handle: str | None = None
    player_explosion_end: float | None = None
    player_explosion_active = False
    player_auto_reset_handle: str | None = None
    learning_stats = {
        "runs": 0,
        "wins": 0,
        "losses": 0,
        "kill_total": 0.0,
        "kill_samples": 0,
        "edge_total": 0.0,
        "edge_samples": 0,
    }
    current_ai_kills = 0
    current_edge_penalty = 0.0
    current_edge_samples = 0

    def _clamp_learning_speed(speed: int) -> int:
        """Keep the learning-speed slider within configured bounds."""

        return max(learning_speed_min, min(learning_speed_max, speed))

    def _current_learning_speed() -> int:
        """Return the currently cached learning-speed multiplier."""
        nonlocal learning_speed_multiplier
        candidate_speed = learning_speed_multiplier
        if learning_speed_var is not None:
            try:
                candidate_speed = int(learning_speed_var.get())
            except Exception:
                candidate_speed = learning_speed_multiplier
        candidate_speed = _clamp_learning_speed(candidate_speed)
        learning_speed_multiplier = candidate_speed
        return candidate_speed

    def _handle_learning_speed_change(*_args: object) -> None:
        """Update the multiplier when the UI spinbox changes."""
        nonlocal learning_speed_multiplier
        if learning_speed_var is None:
            return
            current = learning_speed_var.get()
            try:
                requested_speed = int(current)
            except Exception:
                requested_speed = learning_speed_multiplier
            requested_speed = _clamp_learning_speed(requested_speed)
            learning_speed_multiplier = requested_speed
            if current != requested_speed:
                learning_speed_var.set(requested_speed)

    def _time_scale() -> float:
        """Translate learning mode into a frame-rate multiplier."""
        return _current_learning_speed() if learning_mode else 1.0

    def _tick_steps(scale: float) -> int:
        """Determine how many ticks to run per visual update."""
        if not learning_mode or scale <= 1:
            return 1
        effective_delay = max(
            min_event_delay_ms, base_tick_delay_ms / max(scale, 1.0)
        )
        ticks_per_second = 1000 / effective_delay
        desired_steps_per_second = base_ticks_per_second * scale
        return max(
            1, int(math.ceil(desired_steps_per_second / ticks_per_second))
        )

    def _draw_background() -> None:
        """Paint the night sky gradient and landscape once per launch."""
        gradient_steps = 60
        last_color = "#040912"
        for step in range(gradient_steps):
            ratio = step / gradient_steps
            red = int(5 + ratio * 20)
            green = int(6 + ratio * 25)
            blue = int(12 + ratio * 35)
            color = f"#{red:02x}{green:02x}{blue:02x}"
            last_color = color
            stripe_top = sky_height * (step / gradient_steps)
            stripe_bottom = sky_height * ((step + 1) / gradient_steps)
            canvas.create_rectangle(
                0,
                stripe_top,
                canvas_width,
                stripe_bottom,
                fill=color,
                outline="",
            )
        canvas.create_rectangle(
            0,
            sky_height,
            canvas_width,
            canvas_height,
            fill=last_color,
            outline="",
        )
        for _ in range(120):
            size = random.uniform(1.0, 2.4)
            x = random.randint(0, canvas_width)
            y = random.randint(0, int(sky_height - 60))
            tint = random.randint(200, 255)
            canvas.create_oval(
                x - size,
                y - size,
                x + size,
                y + size,
                fill=f"#{tint:02x}{tint:02x}{255:02x}",
                outline="",
            )
        moon_x = canvas_width * 0.8
        moon_y = sky_height * 0.2
        canvas.create_oval(
            moon_x - 40,
            moon_y - 40,
            moon_x + 40,
            moon_y + 40,
            fill="#f7f3d4",
            outline="#e3deb4",
        )
        canvas.create_oval(
            moon_x - 25,
            moon_y - 35,
            moon_x + 45,
            moon_y + 35,
            fill=last_color,
            outline=last_color,
        )
        ridge_points: list[tuple[float, float]] = []
        segments = 8
        segment_width = canvas_width / segments
        for i in range(segments + 1):
            x = i * segment_width
            offset = random.uniform(-15, 10)
            y = sky_height + offset
            ridge_points.append((x, y))

        def _hill_y_at(x_val: float) -> float:
            """Interpolate the hill height at the given horizontal position."""
            if len(ridge_points) < 2:
                return sky_height
            for (left_x, left_y), (right_x, right_y) in zip(
                ridge_points[:-1], ridge_points[1:]
            ):
                if (left_x <= x_val <= right_x) or (
                    right_x <= x_val <= left_x
                ):
                    if right_x == left_x:
                        return (left_y + right_y) / 2
                    ratio = (x_val - left_x) / (right_x - left_x)
                    return left_y + ratio * (right_y - left_y)
            return ridge_points[-1][1]

        def _draw_skyline() -> None:
            """Draw city skylines over flatter hill intervals."""
            if len(ridge_points) < 2:
                return
            threshold = sky_height + 5
            low_regions: list[tuple[float, float]] = []
            current_start: float | None = None
            current_end: float | None = None
            for x, y in ridge_points:
                if y <= threshold:
                    if current_start is None:
                        current_start = x
                    current_end = x
                elif current_start is not None and current_end is not None:
                    if current_end - current_start >= 70:
                        low_regions.append((current_start, current_end))
                    current_start = None
                    current_end = None
            if current_start is not None and current_end is not None:
                if current_end - current_start >= 70:
                    low_regions.append((current_start, current_end))
            fallback_segments = [
                (canvas_width * 0.05, canvas_width * 0.35),
                (canvas_width * 0.65, canvas_width * 0.95),
                (canvas_width * 0.35, canvas_width * 0.6),
            ]
            if len(low_regions) < 2:
                augmented = low_regions[:]
                for seg in fallback_segments:
                    augmented.append(seg)
                    if len(augmented) >= 2:
                        break
                low_regions = augmented
            selected = sorted(
                low_regions,
                key=lambda pair: pair[1] - pair[0],
                reverse=True,
            )
            building_width = 7
            city_clusters = 0

            def _draw_city_segment(bounds: tuple[float, float]) -> bool:
                """Render one city block within the skyline gap."""
                nonlocal city_clusters
                start_x, end_x = bounds
                region_width = end_x - start_x
                min_width = building_width * 10
                if region_width < min_width:
                    return False
                max_buildings = int(region_width // building_width)
                if max_buildings < 10:
                    return False
                building_count = random.randint(10, min(15, max_buildings))
                offset_space = max(
                    0.0, region_width - building_count * building_width
                )
                base_offset = (
                    0.0
                    if offset_space <= 0
                    else random.uniform(0, offset_space)
                )
                for idx in range(building_count):
                    left = start_x + base_offset + idx * building_width
                    right = left + building_width
                    if right > end_x:
                        break
                    mid_x = (left + right) / 2
                    base_y = _hill_y_at(mid_x)
                    height = random.randint(14, 32)
                    top_y = max(0.0, base_y - height)
                    canvas.create_rectangle(
                        left,
                        top_y,
                        right,
                        base_y + 2,
                        fill="#1b1f2b",
                        outline="#2d3445",
                    )
                    for column_offset in (2, building_width - 3):
                        window_x = left + column_offset - 1
                        window_y = base_y - 4
                        while window_y > top_y + 2:
                            lit = random.random() < 0.35
                            canvas.create_rectangle(
                                window_x,
                                window_y,
                                window_x + 1,
                                window_y + 1,
                                fill="#ffdd7a" if lit else "#4c5d89",
                                outline="",
                            )
                            window_y -= 4
                city_clusters += 1
                return True

            for bounds in selected:
                if city_clusters >= 2:
                    break
                _draw_city_segment(bounds)
            if city_clusters < 2:
                for bounds in fallback_segments:
                    if city_clusters >= 2:
                        break
                    _draw_city_segment(bounds)

        def _draw_trees() -> None:
            """Scatter stylized trees along the hill line."""
            if len(ridge_points) < 2:
                return
            cluster_count = random.randint(3, 6)
            centers = [
                (
                    random.uniform(0, canvas_width),
                    sky_height + random.uniform(-10, 8),
                )
                for _ in range(cluster_count)
            ]
            total_trees = random.randint(30, 50)
            base = total_trees // cluster_count
            remainder = total_trees % cluster_count
            span = canvas_width * 0.05
            for idx, center in enumerate(centers):
                trees_here = base + (1 if idx < remainder else 0)
                center_x = center[0]
                for _ in range(trees_here):
                    x = max(
                        0,
                        min(
                            canvas_width,
                            center_x + random.uniform(-span, span),
                        ),
                    )
                    base_y = _hill_y_at(x)
                    if base_y < sky_height + 2:
                        base_y = sky_height + 2
                    tree_height = random.randint(5, 14)
                    base_width = random.randint(4, 9)
                    trunk_height = random.randint(2, 3)
                    canvas.create_rectangle(
                        x - 0.5,
                        base_y - trunk_height,
                        x + 0.5,
                        base_y + 1,
                        fill="#3a2b1f",
                        outline="#3a2b1f",
                    )
                    canvas.create_polygon(
                        x,
                        base_y - tree_height,
                        x - base_width / 2,
                        base_y - 2,
                        x + base_width / 2,
                        base_y - 2,
                        fill="#1d3f2a",
                        outline="#274c31",
                    )

        def _draw_bushes() -> None:
            """Dot the foreground with brushy shrubs."""
            bush_count = random.randint(20, 38)
            for _ in range(bush_count):
                x = random.uniform(0, canvas_width)
                base_y = _hill_y_at(x)
                radius = random.uniform(0.8, 2.0)
                center_y = base_y - radius * random.uniform(0.2, 0.6)
                top = center_y - radius
                bottom = center_y + radius
                if top < 0:
                    shift = -top
                    top += shift
                    bottom += shift
                    center_y += shift
                canvas.create_oval(
                    x - radius,
                    top,
                    x + radius,
                    bottom,
                    fill="#1f2c1d",
                    outline="#263726",
                )

        _draw_skyline()
        _draw_trees()
        _draw_bushes()
        hill_points = ridge_points + [
            (canvas_width, canvas_height),
            (0, canvas_height),
        ]
        flat_points = [coord for point in hill_points for coord in point]
        canvas.create_polygon(flat_points, fill="#152a1e", outline="#0f1c15")

    def _color_from_hex(hex_color: str) -> tuple[int, int, int]:
        """Convert an HTML hex string into an RGB tuple."""
        return tuple(int(hex_color[slice(i, i + 2)], 16) for i in (1, 3, 5))

    def _color_for_star(base: tuple[int, int, int], brightness: float) -> str:
        """Blend the base color with white according to brightness."""
        brightness = max(0.0, min(1.0, brightness))
        red_component = int(base[0] * brightness + 255 * (1 - brightness))
        green_component = int(base[1] * brightness + 255 * (1 - brightness))
        blue_component = int(base[2] * brightness + 255 * (1 - brightness))
        return f"#{red_component:02x}{green_component:02x}{blue_component:02x}"

    def _clear_shooting_stars() -> None:
        """Remove every scheduled shooting star from the canvas."""
        for star in shooting_stars:
            canvas.delete(star["head"])
            canvas.delete(star["tail"])
        shooting_stars.clear()

    def _schedule_next_shooting_star(multiplier: float = 1.0) -> None:
        """Plan the next shooting-star spawn time."""
        nonlocal next_shooting_star_time
        if learning_mode:
            next_shooting_star_time = float("inf")
            return
        interval = random.expovariate(0.1)
        roll = random.random()
        if roll < 0.2:
            interval *= 0.35
        elif roll > 0.9:
            interval *= 2.5
        interval = interval / _time_scale()
        next_shooting_star_time = time.time() + interval * multiplier

    def _spawn_shooting_star() -> None:
        """Create a shooting star and animate it across the sky."""
        if learning_mode:
            return
        base_colors = ["#fef5d7", "#ffe5b0", "#cde8ff", "#fff0ef"]
        angle = math.radians(random.uniform(15, 165))
        slow = random.random() < 0.5
        if slow:
            speed = random.uniform(2.0, 3.8)
            target_len = random.uniform(140, 200)
        else:
            speed = random.uniform(3.8, 8.0)
            target_len = random.uniform(30, 150)
        start_x = random.uniform(-60, canvas_width + 60)
        start_y = random.uniform(-80, sky_height * 0.4)
        horizontal_velocity = math.cos(angle) * speed
        vertical_velocity = math.sin(angle) * speed
        grow_rate = max(4.0, target_len / random.uniform(6.0, 10.0))
        decay_rate = grow_rate / random.uniform(1.5, 2.5)
        base_color = _color_from_hex(random.choice(base_colors))
        size = random.uniform(3.0, 5.5)
        brightness = 0.25
        head = canvas.create_oval(
            start_x - size,
            start_y - size,
            start_x + size,
            start_y + size,
            fill=_color_for_star(base_color, brightness),
            outline="",
        )
        tail = canvas.create_line(
            start_x,
            start_y,
            start_x,
            start_y,
            fill=_color_for_star(base_color, brightness * 0.8),
            width=1.0,
            capstyle="round",
        )
        shooting_stars.append(
            {
                "head": head,
                "tail": tail,
                "head_x": start_x,
                "head_y": start_y,
                "velocity_x": horizontal_velocity,
                "velocity_y": vertical_velocity,
                "size": size,
                "current_len": 5.0,
                "target_len": target_len,
                "grow_rate": grow_rate,
                "decay_rate": decay_rate,
                "phase": "grow",
                "base_color": base_color,
                "brightness": brightness,
            }
        )
        _schedule_next_shooting_star()

    def _update_shooting_stars() -> None:
        """Advance every shooting star and expire finished ones."""
        nonlocal shooting_stars
        if learning_mode:
            return
        now = time.time()
        while now >= next_shooting_star_time:
            _spawn_shooting_star()
            now = time.time()
        for star in list(shooting_stars):
            speed = math.hypot(star["velocity_x"], star["velocity_y"]) or 1.0
            direction_x = star["velocity_x"] / speed
            direction_y = star["velocity_y"] / speed
            star["head_x"] += star["velocity_x"]
            star["head_y"] += star["velocity_y"]
            if star["phase"] == "grow":
                star["current_len"] = min(
                    star["target_len"],
                    star["current_len"] + star["grow_rate"],
                )
                star["brightness"] = min(1.0, star["brightness"] + 0.08)
                if star["current_len"] >= star["target_len"]:
                    star["phase"] = "decay"
            else:
                star["current_len"] -= star["decay_rate"]
                star["brightness"] = max(0.05, star["brightness"] - 0.04)
            if (
                star["current_len"] <= 0
                or star["head_y"] > canvas_height + 40
                or star["head_x"] < -120
                or star["head_x"] > canvas_width + 120
            ):
                canvas.delete(star["head"])
                canvas.delete(star["tail"])
                shooting_stars.remove(star)
                continue
            tail_x = star["head_x"] - direction_x * star["current_len"]
            tail_y = star["head_y"] - direction_y * star["current_len"]
            canvas.coords(
                star["tail"],
                star["head_x"],
                star["head_y"],
                tail_x,
                tail_y,
            )
            canvas.coords(
                star["head"],
                star["head_x"] - star["size"],
                star["head_y"] - star["size"],
                star["head_x"] + star["size"],
                star["head_y"] + star["size"],
            )
            color = _color_for_star(star["base_color"], star["brightness"])
            canvas.itemconfigure(star["head"], fill=color)
            tail_color = _color_for_star(
                star["base_color"], star["brightness"] * 0.8
            )
            canvas.itemconfigure(star["tail"], fill=tail_color)
            tail_width = max(1.0, star["current_len"] / 80)
            canvas.itemconfigure(star["tail"], width=tail_width)

    _draw_background()
    _schedule_next_shooting_star()
    instructions = tkinter_ttk.Label(
        window,
        text=(
            "Move with the mouse, left-click to fire, "
            "right-click (or ctrl-click) to launch stored "
            "space charges. Catch capsules to stockpile up "
            "to three and watch the shields, charges and "
            "countdown timers."
        ),
        wraplength=canvas_width,
        padding=(0, 4),
    )
    instructions.pack(anchor="w", padx=16, pady=(2, 2))
    run_start_time: float | None = None
    timers_started = False
    kill_order: list[str] = []
    rows_config = [(4, 16), (1, 8), (1, 1)]
    total_rows = sum(rows for rows, _ in rows_config)
    total_enemies = 0
    charge_capacity = max(1, int(charges_cfg.get("capacity", 3)))
    general_shield_max = max(1, int(general_cfg.get("shield", 20)))
    base_general_shield_max = general_shield_max
    general_speed_limit = float(general_cfg.get("max_speed", 7.0))
    player_shield_max = max(1, int(player_cfg.get("shield", 50)))
    fallback_speed = (
        general_speed_limit * 2 if general_speed_limit > 0 else 14.0
    )
    player_speed_limit = float(motion_cfg.get("max_speed", fallback_speed))
    player_speed_limit = max(player_speed_limit, 1.0)
    player_accel = max(0.05, float(motion_cfg.get("accel", 0.75)))
    player_decel = max(0.05, float(motion_cfg.get("decel", 0.65)))
    motion_snap_error = max(0.1, float(motion_cfg.get("snap_error", 0.6)))
    explosion_shard_count = max(1, int(explosion_cfg.get("shard_count", 90)))
    explosion_frame_ms = max(10, int(explosion_cfg.get("frame_ms", 40)))
    explosion_violence = max(
        0.1, float(explosion_cfg.get("violence_scale", 1.0))
    )
    legacy_duration = explosion_cfg.get("duration_seconds")
    player_explosion_hold = float(
        player_explosion_cfg.get(
            "hold_seconds",
            legacy_duration if legacy_duration is not None else 5.0,
        )
    )
    player_explosion_hold = max(0.5, player_explosion_hold)
    debris_default_count = max(1, int(debris_cfg.get("count", 14)))
    debris_damages_all = bool(
        debris_cfg.get("damages_all", debris_cfg.get("damage_enemies", False))
    )
    last_shot_time = 0.0
    autopilot_active = False
    ai_controller = None
    completed_by_ai = False
    status_frame = tkinter_ttk.Frame(window)
    status_frame.pack(fill="x", padx=16, pady=(0, 2))
    shield_status_var = tkinter_module.StringVar()
    tkinter_ttk.Label(
        status_frame,
        textvariable=shield_status_var,
        font=("Helvetica", 15, "bold"),
    ).pack(anchor="w", pady=(0, 2))
    action_var = tkinter_module.StringVar()
    tkinter_ttk.Label(status_frame, textvariable=action_var).pack(
        anchor="w", pady=(0, 4)
    )
    ai_stats_var = tkinter_module.StringVar()
    ai_learning_var = tkinter_module.StringVar()
    kill_meter_var = tkinter_module.StringVar()

    def _format_learning_summary() -> str:
        """Return a short summary string for the learning statistics."""
        runs = learning_stats["runs"]
        wins = learning_stats["wins"]
        losses = learning_stats["losses"]
        total = max(1, wins + losses)
        win_rate = (wins / total) * 100 if total else 0.0
        kill_avg = (
            learning_stats["kill_total"] / learning_stats["kill_samples"]
            if learning_stats["kill_samples"]
            else 0.0
        )
        edge_avg = (
            learning_stats["edge_total"] / learning_stats["edge_samples"]
            if learning_stats["edge_samples"]
            else 0.0
        )
        edge_score = max(0.0, min(1.0, 1.0 - edge_avg)) * 100
        return (
            "Runs trained: {runs}     Win rate: {rate:.0f}%     "
            "Avg kills: {kills:.1f}     Edge discipline: {edge:.0f}% center"
        ).format(
            runs=runs,
            rate=win_rate,
            kills=kill_avg,
            edge=edge_score,
        )

    def _update_ai_stats() -> None:
        """Refresh the AI status labels based on stored learning stats."""
        runs = int(learning_alltime_stats.get("runs", 0))
        saved = int(learning_alltime_stats.get("wins", 0))
        lost = int(learning_alltime_stats.get("losses", 0))
        ai_stats_var.set(
            (
                "AI games: {runs}     Everybody lives: {saved}     "
                "Everybody dies: {lost}"
            ).format(runs=runs, saved=saved, lost=lost)
        )
        ai_learning_var.set(_format_learning_summary())
        _update_kill_meter()

    def _update_kill_meter() -> None:
        """Update the kill counter displayed beneath the AI stats."""
        kill_meter_var.set(f"Kill meter: {current_ai_kills} kills (this run)")

    _update_ai_stats()
    tkinter_ttk.Label(
        status_frame,
        textvariable=ai_stats_var,
        font=("Helvetica", 10, "normal"),
    ).pack(anchor="w", pady=(0, 2))
    tkinter_ttk.Label(
        status_frame,
        textvariable=ai_learning_var,
        font=("Helvetica", 10, "normal"),
    ).pack(anchor="w", pady=(0, 8))
    tkinter_ttk.Label(
        status_frame,
        textvariable=kill_meter_var,
        font=("Helvetica", 10, "normal"),
    ).pack(anchor="w", pady=(0, 8))
    button_bar = tkinter_ttk.Frame(window)
    button_bar.pack(fill="x", padx=16, pady=(2, 12))
    accept_button = tkinter_ttk.Button(
        button_bar, text="Use seed", state=tkinter_module.DISABLED
    )
    accept_button.pack(side="right", padx=(0, 8))
    cancel_button = tkinter_ttk.Button(button_bar, text="Cancel")
    cancel_button.pack(side="right", padx=(0, 8))
    try_again_button = tkinter_ttk.Button(button_bar, text="Reset")
    try_again_button.pack(side="right", padx=(0, 8))

    pause_button = tkinter_ttk.Button(button_bar, text="Pause")
    pause_button.pack(side="left", padx=(0, 8))

    def _show_hall_of_fame() -> None:
        """Open the hall-of-fame overlay for the player."""
        hall_of_fame.show(window)

    def _ai_in_control() -> bool:
        """Return True when the AI has control (learning or autopilot)."""
        return autopilot_active or learning_mode

    def _sample_edge_discipline() -> None:
        """Record how close the ship is to the edges for punishment."""
        nonlocal current_edge_penalty, current_edge_samples
        if not _ai_in_control():
            return
        margin = max(1.0, float(field_margin))
        distance = min(player["x"], canvas_width - player["x"])
        penalty = 0.0
        if distance < margin:
            penalty = min(1.0, (margin - distance) / margin)
        current_edge_penalty += penalty
        current_edge_samples += 1
        if penalty > 0.0:
            ai_brain.penalize_edge(penalty)
        else:
            ai_brain.cool_edge_streak()

    def _finalize_learning_stats(success: bool) -> None:
        """Persist a learning run result and reset temporary counters."""
        nonlocal current_ai_kills, current_edge_penalty, current_edge_samples
        if not _ai_in_control():
            return
        learning_stats["runs"] += 1
        if success:
            learning_stats["wins"] += 1
        else:
            learning_stats["losses"] += 1
        learning_stats["kill_total"] += current_ai_kills
        learning_stats["kill_samples"] += 1
        avg_edge = (
            current_edge_penalty / current_edge_samples
            if current_edge_samples
            else 0.0
        )
        learning_stats["edge_total"] += avg_edge
        learning_stats["edge_samples"] += 1
        learning_alltime_stats["runs"] += 1
        if success:
            learning_alltime_stats["wins"] += 1
        else:
            learning_alltime_stats["losses"] += 1
        learning_alltime_stats["kill_total"] += current_ai_kills
        learning_alltime_stats["kill_samples"] += 1
        learning_alltime_stats["edge_total"] += avg_edge
        learning_alltime_stats["edge_samples"] += 1
        _persist_learning_history()
        _update_ai_stats()
        current_ai_kills = 0
        _update_kill_meter()
        current_edge_penalty = 0.0
        current_edge_samples = 0

    def _reward_enemy_destroyed(record: dict) -> None:
        """Award a reward when the AI destroys an enemy."""
        if not _ai_in_control():
            return
        rank = record.get("rank", "lieutenant")
        ai_brain.reward_enemy_destroyed(
            rank, general=record.get("general", False)
        )
        nonlocal current_ai_kills
        current_ai_kills += 1
        _update_kill_meter()

    def _penalize_enemy_respawned(record: dict) -> None:
        """Apply a penalty when the AI's target respawns."""
        if not _ai_in_control():
            return
        rank = record.get("rank", "lieutenant")
        ai_brain.penalize_enemy_respawned(rank)

    def _cancel_player_auto_reset() -> None:
        """Cancel any scheduled automatic game reset after explosions."""
        nonlocal player_auto_reset_handle
        if player_auto_reset_handle:
            try:
                canvas.after_cancel(player_auto_reset_handle)
            except Exception:
                pass
            player_auto_reset_handle = None

    def _schedule_player_auto_reset() -> None:
        """Schedule an automatic reset after the player explodes."""
        nonlocal player_auto_reset_handle
        if learning_mode:
            return
        _cancel_player_auto_reset()
        delay_ms = max(100, int(player_explosion_hold * 1000))

        def _auto_reset() -> None:
            """Reset the level automatically after a delay."""
            _reset_game()

        player_auto_reset_handle = canvas.after(delay_ms, _auto_reset)

    def _clear_player_explosion() -> None:
        """Remove explosion shards and cancel their animation timer."""
        nonlocal player_explosion_handle
        nonlocal player_explosion_end
        nonlocal player_explosion_active
        for shard in player_explosion:
            canvas.delete(shard["item"])
        player_explosion.clear()
        if player_explosion_handle:
            try:
                canvas.after_cancel(player_explosion_handle)
            except Exception:
                pass
        player_explosion_handle = None
        player_explosion_end = None
        player_explosion_active = False

    def _animate_player_explosion() -> None:
        """Animate the remaining explosion shards until the effect ends."""
        nonlocal player_explosion_handle
        nonlocal player_explosion_active
        if not player_explosion or player_explosion_end is None:
            _clear_player_explosion()
            return
        now = time.time()
        for shard in list(player_explosion):
            canvas.move(
                shard["item"], shard["velocity_x"], shard["velocity_y"]
            )
            shard["velocity_y"] += 0.08 * explosion_violence
            shard["life"] -= 0.015
            coords = canvas.coords(shard["item"])
            if not coords:
                player_explosion.remove(shard)
                continue
            shard_center_x = (coords[0] + coords[2]) / 2
            shard_center_y = (coords[1] + coords[3]) / 2
            size = max(0.5, shard["size"] * shard["life"])
            canvas.coords(
                shard["item"],
                shard_center_x - size,
                shard_center_y - size,
                shard_center_x + size,
                shard_center_y + size,
            )
            if shard["life"] <= 0:
                canvas.delete(shard["item"])
                player_explosion.remove(shard)
        if now >= player_explosion_end or not player_explosion:
            _clear_player_explosion()
            return
        player_explosion_handle = canvas.after(
            explosion_frame_ms,
            _animate_player_explosion,
        )

    def _start_player_explosion() -> None:
        """Spawn explosion fragments and begin their animation."""
        nonlocal player_explosion_handle
        nonlocal player_explosion_end
        nonlocal player_explosion_active
        _clear_player_explosion()
        player_explosion_end = time.time() + player_explosion_hold
        player_explosion_active = True
        canvas.itemconfigure(player_item, state="hidden")
        colors = ["#ffd166", "#ff8a5b", "#ff4d6d", "#ffe29a", "#ffb347"]
        size_scale = max(0.5, min(explosion_violence, 2.0))
        for _ in range(explosion_shard_count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(2.5, 6.5) * explosion_violence
            size = random.uniform(3.5, 8.5) * size_scale
            horizontal_velocity = math.cos(angle) * speed
            vertical_velocity = math.sin(angle) * speed
            color = random.choice(colors)
            fragment_shape_id = canvas.create_oval(
                player["x"] - size,
                player["y"] - size,
                player["x"] + size,
                player["y"] + size,
                fill=color,
                outline="",
            )
            player_explosion.append(
                {
                    "item": fragment_shape_id,
                    "velocity_x": horizontal_velocity,
                    "velocity_y": vertical_velocity,
                    "size": size,
                    "life": random.uniform(0.8, 1.2),
                }
            )
        player_explosion_handle = canvas.after(
            explosion_frame_ms,
            _animate_player_explosion,
        )

    hall_button = tkinter_ttk.Button(
        button_bar, text="Hall of fame", command=_show_hall_of_fame
    )
    hall_button.pack(side="left", padx=(0, 8))

    def _toggle_ai_pilot() -> None:
        """Enable or disable the autopilot controller."""
        if ai_controller.running:
            ai_controller.stop()
            action_var.set("AI disengaged. Manual control restored.")
        else:
            ai_controller.start()

    ai_button = tkinter_ttk.Button(
        button_bar, text="Let AI take care", command=_toggle_ai_pilot
    )
    ai_button.pack(side="left", padx=(0, 8))

    def _toggle_learning() -> None:
        """Start or stop the continuous learning loop."""
        nonlocal learning_mode
        if learning_mode:
            learning_mode = False
            learning_button.configure(text="Let AI learn")
            if ai_controller.running:
                ai_controller.stop()
            ai_button.state(["!disabled"])
            action_var.set("Learning loop stopped.")
            _schedule_next_shooting_star()
        else:
            _clear_shooting_stars()
            learning_mode = True
            learning_button.configure(text="Stop learning")
            ai_button.state(["disabled"])
            if not ai_controller.running:
                ai_controller.start()
            action_var.set("AI is now running continuous practice rounds.")

    learning_button = tkinter_ttk.Button(
        button_bar, text="Let AI learn", command=_toggle_learning
    )
    learning_button.pack(side="left", padx=(0, 8))

    if tkinter_module is not None and tkinter_ttk is not None:
        learning_speed_var = tkinter_module.IntVar(
            value=learning_speed_multiplier
        )
        learning_speed_var.trace_add("write", _handle_learning_speed_change)
        tkinter_ttk.Label(button_bar, text="Learning speed").pack(
            side="left", padx=(8, 4)
        )
        learning_speed_spin = tkinter_ttk.Spinbox(
            button_bar,
            from_=learning_speed_min,
            to=learning_speed_max,
            width=4,
            textvariable=learning_speed_var,
            justify="center",
            command=_handle_learning_speed_change,
        )
        learning_speed_spin.pack(side="left", padx=(0, 8))
    else:
        learning_speed_var = None

    def _perform_ai_forget() -> None:
        """Wipe the AI history and learning stats for a fresh start."""
        ai_brain.forget()
        _reset_learning_history()
        _update_ai_stats()
        action_var.set("AI memory wiped. Fresh slate!")

    def _request_forget() -> None:
        """Prompt the user before erasing the AI memory."""
        if tkinter_module is None:
            _perform_ai_forget()
            return
        dialog = tkinter_module.Toplevel(window)
        dialog.title("Let AI forget?")
        dialog.resizable(False, False)
        tkinter_ttk.Label(
            dialog,
            text="Are you sure you will wipe the poor fella's memory?",
            wraplength=320,
            padding=12,
        ).pack(fill="x")
        btn_row = tkinter_ttk.Frame(dialog)
        btn_row.pack(padx=12, pady=(0, 12))

        def _wipe() -> None:
            """Perform the AI memory wipe after confirmation."""
            _perform_ai_forget()
            dialog.destroy()

        tkinter_ttk.Button(btn_row, text="Wipe", command=_wipe).pack(
            side="left", padx=(0, 8)
        )
        tkinter_ttk.Button(
            btn_row, text="Pardon", command=dialog.destroy
        ).pack(side="left", padx=(0, 8))
        dialog.transient(window)
        dialog.grab_set()
        dialog.protocol("WM_DELETE_WINDOW", dialog.destroy)

    forget_button = tkinter_ttk.Button(
        button_bar, text="Let AI forget", command=_request_forget
    )
    forget_button.pack(side="left", padx=(0, 8))

    def _toggle_pause() -> None:
        """Pause or resume the gameplay loop."""
        nonlocal paused, timers_started
        if game_over:
            return
        if not timers_started and not paused:
            return
        paused = not paused
        if paused:
            pause_button.configure(text="Resume")
            _cancel_timers()
            action_var.set("Paused. Click Resume to continue defending Earth.")
            timers_started = False
        else:
            pause_button.configure(text="Pause")
            _start_timers()
            action_var.set("Resumed. Keep firing!")
            timers_started = True

    pause_button.configure(command=_toggle_pause)
    player_height = 22

    def _player_shape_coords(center_x: float, center_y: float) -> list[float]:
        """Return the polygon coordinates for the player sprite."""
        return [
            center_x,
            center_y - 22,
            center_x + 10,
            center_y - 10,
            center_x + 14,
            center_y - 2,
            center_x + 10,
            center_y + 8,
            center_x + 4,
            center_y + 16,
            center_x - 4,
            center_y + 16,
            center_x - 10,
            center_y + 8,
            center_x - 14,
            center_y - 2,
            center_x - 10,
            center_y - 10,
        ]

    player = {
        "x": canvas_width / 2,
        "y": canvas_height - ground_height - 60,
    }
    player_item = canvas.create_polygon(
        *_player_shape_coords(player["x"], player["y"]),
        fill="#54d1ff",
        outline="#e0f5ff",
        width=2,
    )
    player_last_x = player["x"]
    player_idle_ticks = 0
    player_velocity = 0.0
    player_target_x = player["x"]
    PLAYER_LINGER_THRESHOLD = 60

    def _set_player_target(target_x: float, *, snap: bool = False) -> None:
        """Update the desired horizontal target for the player sprite."""
        nonlocal player_target_x, player_velocity
        clamped = max(40, min(canvas_width - 40, target_x))
        player_target_x = clamped
        if snap:
            player["x"] = clamped
            player_velocity = 0.0
            canvas.coords(
                player_item,
                *_player_shape_coords(player["x"], player["y"]),
            )

    def _update_player_motion() -> None:
        """Move the player toward the target position respecting accel."""
        nonlocal player_velocity
        if game_over or paused:
            return
        delta = player_target_x - player["x"]
        if abs(delta) <= motion_snap_error:
            if player["x"] != player_target_x:
                player["x"] = player_target_x
                canvas.coords(
                    player_item,
                    *_player_shape_coords(player["x"], player["y"]),
                )
            player_velocity = 0.0
            return
        desired_velocity = max(
            -player_speed_limit, min(player_speed_limit, delta)
        )
        step = (
            player_accel
            if abs(desired_velocity) > abs(player_velocity)
            else player_decel
        )
        step = max(step, 0.01)
        change = max(-step, min(step, desired_velocity - player_velocity))
        player_velocity += change
        player_velocity = max(
            -player_speed_limit, min(player_speed_limit, player_velocity)
        )
        player["x"] += player_velocity
        player["x"] = max(40, min(canvas_width - 40, player["x"]))
        if (
            player["x"] in (40, canvas_width - 40)
            and abs(player_velocity) < step * 1.2
        ):
            player_velocity = 0.0
        canvas.coords(
            player_item,
            *_player_shape_coords(player["x"], player["y"]),
        )

    player_shots: list[dict] = []
    enemy_shots: list[dict] = []
    charges: list[dict] = []
    bombs: list[dict] = []
    debris: list[dict] = []
    enemy_data: dict[str, dict] = {}
    general_id: str | None = None
    general_ai = {
        "target": None,
        "mode": "patrol",
        "cooldown": 0.0,
        "velocity_x": 0.0,
        "retreat": False,
        "retreat_target": None,
    }
    charge_count = 0
    destroyed_stack: list[str] = []
    pending_order: list[str] | None = None
    pending_duration = 0.0
    game_over = False
    last_shooter: str | None = None
    general_hits = 0

    def _minions_alive() -> bool:
        """Return True when enemies besides the general still survive."""

        return any(
            record.get("alive") and eid != general_id
            for eid, record in enemy_data.items()
        )

    def _effective_general_shield_max() -> int:
        """Lower the general's shield when it is the only threat left."""

        return 1 if not _minions_alive() else base_general_shield_max

    player_hits = 0
    tick_handle: str | None = None
    fire_handle: str | None = None
    charge_handle: str | None = None
    general_fire_handle: str | None = None
    general_barrage_cooldown = 0

    def _update_enemy_shield_visual(enemy_id: str) -> None:
        """Refresh the outline/width for the enemy shield indicator."""
        record = enemy_data.get(enemy_id)
        if not record:
            return
        rank = record.get("rank", "lieutenant")
        outline = ""
        width = 0
        if rank == "colonel":
            ratio = record["health_points"] / record["max_health_points"]
            outline = "#faf08c"
            if ratio > 0.8:
                width = 4
            elif ratio > 0.6:
                width = 3
            elif ratio > 0.4:
                width = 2
            elif ratio > 0.2:
                width = 1
        elif rank == "major":
            outline = "#a8e3ff"
            width = 2 if record["health_points"] > 1 else 0
        elif rank == "general":
            outline = "#ffdbe8"
            width = 3
        canvas.itemconfigure(
            record["item"],
            outline=outline if width else "",
            width=width,
        )

    def _enemy_polygon(
        x: float, y: float, *, general: bool = False, elite: bool = False
    ) -> int:
        """Return a Tk polygon id for the given enemy type and position."""

        if general:
            return canvas.create_polygon(
                x - 18,
                y + 10,
                x,
                y - 18,
                x + 18,
                y + 10,
                x + 10,
                y + 18,
                x - 10,
                y + 18,
                fill="#ff679d",
                outline="#ffdbe8",
            )
        base_color = "#ffb347"
        outline_color = "#ffe0b3"
        if elite:
            base_color = "#ffec8b"
            outline_color = "#f7d45a"
        return canvas.create_polygon(
            x - 14,
            y + 8,
            x - 6,
            y - 8,
            x + 6,
            y - 8,
            x + 14,
            y + 8,
            x,
            y + 14,
            fill=base_color,
            outline=outline_color,
        )

    def _spawn_enemies() -> None:
        """Populate each wave of enemies before a run begins."""
        nonlocal general_id, general_ai, total_enemies
        general_id = None
        row_gap = 70
        start_y = 80
        eid = 0
        row_counter = 0
        total_groups = len(rows_config)
        general_rows = sum(
            rows
            for idx, (rows, cols) in enumerate(rows_config)
            if idx == total_groups - 1 and cols == 1
        )
        non_general_total_rows = max(0, total_rows - general_rows)
        non_general_counter = 0
        for group_index, (rows, cols) in enumerate(rows_config):
            general_group = group_index == total_groups - 1 and cols == 1
            for _ in range(rows):
                row_index = row_counter
                row_cols = cols
                stagger_row = (
                    not general_group and row_index == 1 and cols >= 2
                )
                if stagger_row:
                    row_cols = max(1, cols - 1)
                if row_cols > 1:
                    spacing = (canvas_width - 2 * field_margin) / max(
                        row_cols - 1, 1
                    )
                    total_span = spacing * (row_cols - 1)
                    start_x = (canvas_width - total_span) / 2
                else:
                    spacing = 0
                    start_x = canvas_width / 2
                if stagger_row and row_cols > 1:
                    start_x += spacing / 2
                y = start_y + row_index * row_gap
                row_counter += 1
                non_general_index: int | None = None
                if not general_group:
                    non_general_index = non_general_counter
                    non_general_counter += 1
                is_bottom = (
                    non_general_index is not None
                    and non_general_total_rows
                    and non_general_index == non_general_total_rows - 1
                )
                is_major = (
                    non_general_index is not None
                    and non_general_total_rows > 1
                    and non_general_index == non_general_total_rows - 2
                )
                row_positions: list[float] = []
                for column_index in range(row_cols):
                    general = general_group
                    x = start_x + column_index * spacing
                    if (
                        stagger_row
                        and row_cols > 1
                        and column_index == row_cols - 1
                        and not general
                    ):
                        continue
                    eid += 1
                    if not general:
                        x = max(
                            field_margin, min(canvas_width - field_margin, x)
                        )
                        row_positions.append(x)
                if general:
                    rank = "general"
                    max_health_points = base_general_shield_max
                elif is_bottom:
                    rank = "colonel"
                    max_health_points = 5
                elif is_major:
                    rank = "major"
                    max_health_points = 2
                else:
                    rank = "lieutenant"
                    max_health_points = 1
                health_points = max_health_points
                enemy_shape_id = _enemy_polygon(
                    x, y, general=general, elite=is_bottom or is_major
                )
                enemy_id = f"E{eid:02d}"
                enemy_data[enemy_id] = {
                    "item": enemy_shape_id,
                    "x": x,
                    "y": y,
                    "spawn_x": x,
                    "spawn_y": y,
                    "alive": True,
                    "general": general,
                    "health_points": health_points,
                    "max_health_points": max_health_points,
                    "rank": rank,
                    "rail_y": y if general else None,
                }
                if general:
                    general_id = enemy_id
                    general_ai = {
                        "target": canvas_width - field_margin,
                        "mode": "patrol",
                        "cooldown": random.randint(60, 120),
                        "velocity_x": 0.0,
                    }
                _update_enemy_shield_visual(enemy_id)
                if stagger_row and not general_group and row_positions:
                    if len(row_positions) >= 2:
                        delta = row_positions[1] - row_positions[0]
                    else:
                        delta = spacing or (
                            canvas_width - 2 * field_margin
                        ) / max(cols - 1, 1)
                    delta = max(delta, 20)
                    extensions = [
                        row_positions[0] - delta,
                        row_positions[-1] + delta,
                    ]
                    for offset in (-1, 1):
                        extra_x = (
                            extensions[0] if offset < 0 else extensions[1]
                        )
                        extra_x = max(10, min(canvas_width - 10, extra_x))
                        eid += 1
                        x = extra_x
                        if is_bottom:
                            rank = "colonel"
                            max_health_points = 5
                        elif is_major:
                            rank = "major"
                            max_health_points = 2
                        else:
                            rank = "lieutenant"
                            max_health_points = 1
                        health_points = max_health_points
                        staggered_enemy_shape_id = _enemy_polygon(
                            x, y, general=False, elite=is_bottom or is_major
                        )
                        enemy_id = f"E{eid:02d}"
                        enemy_data[enemy_id] = {
                            "item": staggered_enemy_shape_id,
                            "x": x,
                            "y": y,
                            "spawn_x": x,
                            "spawn_y": y,
                            "alive": True,
                            "general": False,
                            "health_points": health_points,
                            "max_health_points": max_health_points,
                            "rank": rank,
                            "rail_y": None,
                        }
                        _update_enemy_shield_visual(enemy_id)
        total_enemies = len(enemy_data)

    def _scaled_after(delay_ms: int, callback: Callable[[], None]) -> str:
        """Schedule a callback factoring in the current time scale."""
        scale = _time_scale()
        scaled = max(min_event_delay_ms, int(delay_ms / max(scale, 1.0)))
        return canvas.after(scaled, callback)

    MAX_RUN_SECONDS = max_run_seconds

    def _elapsed_seconds() -> int:
        """Return the scaled seconds elapsed since the game started."""
        if run_start_time is None:
            return 0
        return max(0, int((time.time() - run_start_time) * _time_scale()))

    def _time_left_seconds() -> int:
        """Compute how many scaled seconds remain in the current round."""
        return max(0, MAX_RUN_SECONDS - _elapsed_seconds())

    def _format_time_left() -> str:
        """Format the remaining time as minutes and seconds."""
        remaining = _time_left_seconds()
        minutes, seconds = divmod(remaining, 60)
        return f"{minutes:02d}:{seconds:02d}"

    def _update_status() -> None:
        """Refresh shield, charge and timer labels in the status bar."""
        player_remaining = max(player_shield_max - player_hits, 0)
        current_general_shield = _effective_general_shield_max()
        general_remaining = max(current_general_shield - general_hits, 0)
        gap_primary = " " * 10
        gap_secondary = " " * 12
        gap_timer = " " * 10
        time_left = _time_left_seconds()
        urgency = ""
        if time_left <= 120 and not game_over:
            urgency = f"    T-{time_left:03d}s"
        shield_status_var.set(
            f"Your shield: {player_remaining}/{player_shield_max}"
            f"{gap_primary}"
            f"General shield: {general_remaining}/{current_general_shield}"
            f"{gap_secondary}Neutron charges: {charge_count}/{charge_capacity}"
            f"{gap_timer}Time left: {_format_time_left()}{urgency}"
        )

    def _record_ai_outcome(
        success: bool, duration: float, *, controlling: bool | None = None
    ) -> None:
        """Log what happened to the AI after a run completes."""
        if controlling is None:
            controlling = _ai_in_control()
        if not controlling:
            return
        ai_brain.record_run(success=success, duration=duration)
        _finalize_learning_stats(success)
        _update_ai_stats()

    def _cancel_timers() -> None:
        """Cancel every recurring timer task currently scheduled."""
        nonlocal tick_handle, fire_handle, charge_handle, general_fire_handle
        for handle in (
            tick_handle,
            fire_handle,
            charge_handle,
            general_fire_handle,
        ):
            if handle:
                try:
                    canvas.after_cancel(handle)
                except Exception:
                    pass
        tick_handle = fire_handle = charge_handle = general_fire_handle = None

    def _close_window() -> None:
        """Shut down the game window and stop all background tasks."""
        nonlocal learning_mode
        _cancel_learning_restart()
        _cancel_player_auto_reset()
        _cancel_timers()
        learning_mode = False
        if ai_controller:
            ai_controller.stop()
        _clear_player_explosion()
        window.destroy()

    def _cancel_learning_restart() -> None:
        """Cancel the pending learning restart timer if active."""
        nonlocal learning_restart_handle
        if learning_restart_handle:
            try:
                canvas.after_cancel(learning_restart_handle)
            except Exception:
                pass
            learning_restart_handle = None

    def _center_player() -> None:
        """Reposition and reorient the player to the centre of the field."""
        nonlocal player_last_x, player_idle_ticks
        player["y"] = canvas_height - ground_height - 60
        _set_player_target(canvas_width / 2, snap=True)
        player_last_x = player["x"]
        player_idle_ticks = 0

    def _player_lingering() -> bool:
        """Return True once the player has stayed idle long enough."""
        return player_idle_ticks >= PLAYER_LINGER_THRESHOLD

    def _reset_game(*, preserve_ai: bool = False) -> None:
        """Reset the game state while optionally keeping the AI learning
        loop alive."""
        nonlocal run_start_time, kill_order, charge_count
        nonlocal general_hits, general_ai, timers_started
        nonlocal player_shots, enemy_shots, charges, bombs, destroyed_stack
        nonlocal pending_order, pending_duration, game_over, debris
        nonlocal player_hits, autopilot_active, ai_controller, completed_by_ai
        nonlocal paused, learning_mode, player_velocity, player_target_x
        nonlocal current_ai_kills, current_edge_penalty, current_edge_samples
        _cancel_learning_restart()
        if paused:
            paused = False
            pause_button.configure(text="Pause")
        _cancel_timers()
        if learning_mode and not preserve_ai:
            learning_mode = False
            learning_button.configure(text="Let AI learn")
            ai_button.state(["!disabled"])
        for record in enemy_data.values():
            canvas.delete(record["item"])
        enemy_data.clear()
        for projectile in player_shots + enemy_shots + bombs:
            canvas.delete(projectile["item"])
        for charge in charges:
            canvas.delete(charge["item"])
        for fragment in debris:
            canvas.delete(fragment["item"])
        player_shots.clear()
        enemy_shots.clear()
        bombs.clear()
        charges.clear()
        debris.clear()
        _clear_player_explosion()
        canvas.itemconfigure(player_item, state="normal")
        for star in shooting_stars:
            canvas.delete(star["head"])
            canvas.delete(star["tail"])
        shooting_stars.clear()
        _schedule_next_shooting_star()
        destroyed_stack.clear()
        kill_order = []
        general_ai = {
            "target": None,
            "mode": "patrol",
            "cooldown": 0.0,
            "velocity_x": 0.0,
        }
        timers_started = False
        run_start_time = None
        charge_count = 0
        general_hits = 0
        pending_order = None
        pending_duration = 0.0
        game_over = False
        player_hits = 0
        player_velocity = 0.0
        player_target_x = canvas_width / 2
        current_ai_kills = 0
        current_edge_penalty = 0.0
        current_edge_samples = 0
        completed_by_ai = False
        restart_ai = preserve_ai and ai_controller and ai_controller.running
        if ai_controller and ai_controller.running:
            ai_controller.stop()
        _spawn_enemies()
        ai_brain.begin_run()
        _center_player()
        _update_status()
        action_var.set("Click the field to start the invasion.")
        if learning_mode and preserve_ai:
            action_var.set("AI launching the next practice round.")
        accept_button.state(["disabled"])
        if restart_ai and learning_mode:
            ai_controller.start()

    def _schedule_learning_restart() -> None:
        """Queue another learning round once the AI finishes its sortie."""
        nonlocal learning_restart_handle
        if not learning_mode:
            return
        _cancel_learning_restart()
        action_var.set("AI reviewing the battle log. Next sortie incoming...")

        def _restart() -> None:
            """Restart the run without resetting the AI state."""
            _reset_game(preserve_ai=True)

        learning_restart_handle = _scaled_after(1200, _restart)

    def _respawn_enemy(enemy_id: str, *, opposite: bool = False) -> None:
        """Bring a destroyed enemy back to life for revives."""
        record = enemy_data.get(enemy_id)
        if not record or record["alive"]:
            return
        spawn_x = record.get("spawn_x", record.get("x", canvas_width / 2))
        spawn_y = record.get("spawn_y", record.get("y", 80))
        record["x"] = spawn_x
        record["y"] = spawn_y
        record["health_points"] = record["max_health_points"]
        elite = record["rank"] in ("colonel", "major")
        record["item"] = _enemy_polygon(
            record["x"], record["y"], general=record["general"], elite=elite
        )
        record["alive"] = True
        if record["general"]:
            record["rail_y"] = record["y"]
        _update_enemy_shield_visual(enemy_id)
        _penalize_enemy_respawned(record)

    def _pick_revive_target() -> str | None:
        """Choose the next destroyed enemy that should revive."""
        if not destroyed_stack:
            return None
        priorities = ("lieutenant", "major", "colonel", "general")
        for rank in priorities:
            for enemy_id in reversed(destroyed_stack):
                record = enemy_data.get(enemy_id)
                if (
                    record
                    and record.get("rank") == rank
                    and not record["alive"]
                ):
                    destroyed_stack.remove(enemy_id)
                    return enemy_id
        return destroyed_stack.pop()

    def _player_defeated() -> None:
        """Handle the player losing their shields and ending the run."""
        nonlocal game_over, autopilot_active
        if game_over:
            return
        _cancel_timers()
        _cancel_player_auto_reset()
        accept_button.state(["disabled"])
        game_over = True
        duration = 0.0
        if run_start_time is not None:
            duration = (time.time() - run_start_time) * _time_scale()
        controlling = _ai_in_control()
        _record_ai_outcome(
            success=False, duration=duration, controlling=controlling
        )
        if learning_mode:
            action_var.set("Training run failed. Relaunching immediately.")
            _reset_game(preserve_ai=True)
            return
        _start_player_explosion()
        action_var.set(
            "Your shield collapsed! Reset or wait for auto-restart."
        )
        autopilot_active = False
        if ai_controller:
            ai_controller.stop()
        _schedule_player_auto_reset()

    def _handle_player_hit() -> None:
        """Process when the player takes damage and update the UI."""
        nonlocal player_hits
        if game_over:
            return
        player_hits = min(player_shield_max, player_hits + 1)
        _update_status()
        if player_hits >= player_shield_max:
            action_var.set(
                "Your shield collapsed! Reset now or wait a moment."
            )
            _player_defeated()
            return
        revived = _pick_revive_target()
        if revived:
            _respawn_enemy(revived, opposite=True)
            if revived in kill_order:
                kill_order.remove(revived)
            action_var.set(
                "You were hit! An invader regrouped. Keep fighting."
            )
        else:
            action_var.set("You were hit, but the fleet holds steady!")

    def _destroy_enemy(enemy_id: str, explosion: bool = False) -> None:
        """Mark an enemy as destroyed and update the fight status."""
        nonlocal pending_order, pending_duration, game_over, general_hits
        nonlocal completed_by_ai, current_ai_kills
        record = enemy_data.get(enemy_id)
        if not record or not record["alive"]:
            return
        if record["general"]:
            remaining = any(
                rec["alive"] and not rec["general"]
                for rec in enemy_data.values()
            )
            if remaining and general_hits < _effective_general_shield_max():
                general_hits += 1
                action_var.set(
                    "The general deflects the blast "
                    f"({general_hits}/{_effective_general_shield_max()})."
                )
                _update_status()
                return
        elif record.get("health_points", 1) > 1 and not explosion:
            record["health_points"] -= 1
            remaining_hp = record["health_points"]
            _update_enemy_shield_visual(enemy_id)
            descriptor = (
                "Colonel"
                if record.get("rank") == "colonel"
                else "Major" if record.get("rank") == "major" else "Cruiser"
            )
            action_var.set(
                f"{descriptor} absorbed the hit ({remaining_hp}/"
                f"{record['max_health_points']} shields)."
            )
            return
        canvas.delete(record["item"])
        record["alive"] = False
        if _ai_in_control():
            current_ai_kills += 1
        _reward_enemy_destroyed(record)
        destroyed_stack.append(enemy_id)
        kill_order.append(enemy_id)
        if explosion:
            action_var.set(
                "Space charge unleashed a chain reaction! Nice shot."
            )
            _spawn_debris(record["x"], record["y"], speed_scale=1.6)
        if record["general"]:
            general_hits = _effective_general_shield_max()
        _update_status()
        if len(kill_order) >= total_enemies and total_enemies > 0:
            pending_order = kill_order[:]
            if run_start_time is not None:
                pending_duration = (
                    time.time() - run_start_time
                ) * _time_scale()
            else:
                pending_duration = 0.0
            controlling = _ai_in_control()
            completed_by_ai = controlling
            if learning_mode:
                action_var.set("Fleet neutralised! AI logging the victory.")
                accept_button.state(["disabled"])
            else:
                action_var.set(
                    "Fleet neutralised! Click Use seed to apply the result."
                )
                accept_button.state(["!disabled"])
            game_over = True
            initial = "AI" if completed_by_ai else "NI"
            time_left_for_record = max(0.0, MAX_RUN_SECONDS - pending_duration)
            hall_of_fame.record(initial, time_left_for_record)
            _record_ai_outcome(
                success=True,
                duration=pending_duration,
                controlling=controlling,
            )
            _cancel_timers()
            if ai_controller and not learning_mode:
                ai_controller.stop()
            _schedule_learning_restart()

    def _fire_player_shot(_event: "tkinter_module.Event | None") -> None:
        """Launch a player laser shot if firing rate and state allow."""
        nonlocal last_shot_time
        if len(player_shots) > 4 or game_over or paused:
            return
        now = time.perf_counter()
        if now - last_shot_time < 0.1:
            return
        _ensure_game_started()
        last_shot_time = now
        player_shot_id = canvas.create_rectangle(
            player["x"] - 2,
            player["y"] - player_height / 2,
            player["x"] + 2,
            player["y"] - player_height / 2 - 12,
            fill="#ffffff",
            outline="",
        )
        player_shots.append({"item": player_shot_id, "velocity_y": -12})

    def _ensure_game_started() -> None:
        """Start the tick/shot timers once the player acts."""
        nonlocal timers_started, run_start_time
        if timers_started:
            return
        timers_started = True
        run_start_time = time.time()
        _start_timers()

    def _move_player_to(target_x: float, *, snap: bool = False) -> None:
        """Set a movement target for the player, ignoring game-over events."""
        if game_over:
            return
        _set_player_target(target_x, snap=snap)

    def _move_player(event: "tkinter_module.Event") -> None:
        """Handle mouse movement to steer the player sprite."""
        if game_over or paused:
            return
        _set_player_target(event.x)

    def _fire_enemy_shot(
        enemy_id: str, *, aim_for: float | None = None
    ) -> bool:
        """Have the specified enemy fire a projectile toward the player."""
        record = enemy_data.get(enemy_id)
        if not record or not record["alive"]:
            return False
        vertical_velocity = 6
        horizontal_velocity = 0.0
        if record.get("rank") == "general":
            vertical_velocity = 5
        enemy_projectile_id = canvas.create_rectangle(
            record["x"] - 4,
            record["y"] + 10,
            record["x"] + 4,
            record["y"] + 22,
            fill="#ff3366",
            outline="",
        )
        enemy_shots.append(
            {
                "item": enemy_projectile_id,
                "velocity_y": vertical_velocity,
                "velocity_x": horizontal_velocity,
                "owner": enemy_id,
            }
        )
        return True

    def _fire_weight(enemy_id: str) -> float:
        """Return a weight that biases enemy shot selection."""
        record = enemy_data.get(enemy_id)
        if not record:
            return 1.0
        rank = record.get("rank")
        if rank == "colonel":
            return 3.5
        if rank == "major":
            return 2.5
        if rank == "general":
            return 4.0
        return 1.0

    def _spawn_enemy_shot_once() -> None:
        """Have a subset of enemies fire once per cycle."""
        nonlocal last_shooter
        live_enemies = [eid for eid, rec in enemy_data.items() if rec["alive"]]
        if not live_enemies:
            return
        per_cycle = (
            1
            if len(live_enemies) < 3
            else random.randint(1, min(3, len(live_enemies)))
        )
        fired_ids: set[str] = set()
        for _ in range(per_cycle):
            candidates = [eid for eid in live_enemies if eid not in fired_ids]
            if not candidates:
                break
            if last_shooter in candidates and len(candidates) > 1:
                candidates = [eid for eid in candidates if eid != last_shooter]
            if not candidates:
                break
            total_weight = sum(_fire_weight(eid) for eid in candidates)
            pick = random.uniform(0, total_weight)
            cumulative = 0.0
            shooter_id = candidates[-1]
            for candidate in candidates:
                cumulative += _fire_weight(candidate)
                if cumulative >= pick:
                    shooter_id = candidate
                    break
            if _fire_enemy_shot(shooter_id):
                fired_ids.add(shooter_id)
                last_shooter = shooter_id

    def _spawn_charge_once() -> None:
        """Spawn an extra space charge capsule when permitted."""
        if len(charges) >= 2 or charge_count >= charge_capacity:
            return
        x = random.randint(60, canvas_width - 60)
        charge_capsule_id = canvas.create_oval(
            x - 8, 30, x + 8, 46, fill="#b0f3ff", outline="#68d4ff"
        )
        charges.append({"item": charge_capsule_id, "velocity_y": 1.5})

    def _enemy_fire_cycle() -> None:
        """Timer callback that periodically triggers enemy shots."""
        nonlocal fire_handle
        if game_over:
            return
        if paused:
            fire_handle = _scaled_after(400, _enemy_fire_cycle)
            return
        _spawn_enemy_shot_once()
        interval = random.randint(600, 1200)
        fire_handle = _scaled_after(interval, _enemy_fire_cycle)

    def _charge_cycle() -> None:
        """Timer callback that spawns charges at fixed intervals."""
        nonlocal charge_handle
        if game_over:
            return
        if paused:
            charge_handle = _scaled_after(1000, _charge_cycle)
            return
        _spawn_charge_once()
        charge_handle = _scaled_after(6000, _charge_cycle)

    def _general_fire_cycle() -> None:
        """Handle the general's occasional barrage."""
        nonlocal general_fire_handle, general_barrage_cooldown
        if game_over:
            return
        if paused:
            general_fire_handle = _scaled_after(600, _general_fire_cycle)
            return
        interval = random.randint(600, 1200)
        if general_id and general_barrage_cooldown <= 0:
            bursts = random.randint(1, 3)
            for _ in range(bursts):
                _fire_enemy_shot(general_id, aim_for=player["x"])
            general_barrage_cooldown = random.randint(2200, 3600)
        else:
            general_barrage_cooldown = max(
                0, general_barrage_cooldown - interval
            )
        general_fire_handle = _scaled_after(interval, _general_fire_cycle)

    def _launch_bomb() -> None:
        """Launch a stored space charge bomb from the player."""
        bomb_shape_id = canvas.create_polygon(
            player["x"] - 6,
            player["y"] - 10,
            player["x"],
            player["y"] - 30,
            player["x"] + 6,
            player["y"] - 10,
            fill="#7dd9ff",
            outline="#bdefff",
        )
        bombs.append({"item": bomb_shape_id, "velocity_y": -1.5})

    def _handle_right_click(
        event: "tkinter_module.Event | None", *, announce: bool = True
    ) -> None:
        """Handle right-clicking to fire a stored space charge."""
        nonlocal charge_count
        if game_over or paused:
            return
        _ensure_game_started()
        if charge_count > 0:
            _launch_bomb()
            charge_count -= 1
            if announce:
                action_var.set(
                    "Space charge launched! "
                    f"{charge_count}/{charge_capacity} remain."
                )
            _update_status()
        else:
            if announce:
                action_var.set("No stored space charge yet—catch a capsule.")

    def _check_enemy_collision(item_id: int) -> str | None:
        """Return the enemy id overlapping the given canvas item."""
        bbox = canvas.bbox(item_id)
        if not bbox:
            return None
        overlaps = canvas.find_overlapping(*bbox)
        for eid, record in enemy_data.items():
            if record["alive"] and record["item"] in overlaps:
                return eid
        return None

    def _rects_overlap(
        first_bbox: tuple[int, int, int, int],
        second_bbox: tuple[int, int, int, int],
    ) -> bool:
        """Return True if two bounding boxes intersect."""
        return not (
            first_bbox[2] <= second_bbox[0]
            or first_bbox[0] >= second_bbox[2]
            or first_bbox[3] <= second_bbox[1]
            or first_bbox[1] >= second_bbox[3]
        )

    def _check_projectile_overlap(
        item_id: int, projectiles: list[dict]
    ) -> dict | None:
        """Return a projectile that overlaps the supplied item, if any."""
        bbox = canvas.bbox(item_id)
        if not bbox:
            return None
        for projectile in projectiles:
            other_bbox = canvas.bbox(projectile["item"])
            if other_bbox and _rects_overlap(bbox, other_bbox):
                return projectile
        return None

    def _check_debris_collision(item_id: int) -> dict | None:
        """Return a debris fragment that overlaps the item, if any."""
        bbox = canvas.bbox(item_id)
        if not bbox:
            return None
        for fragment in debris:
            other_bbox = canvas.bbox(fragment["item"])
            if other_bbox and _rects_overlap(bbox, other_bbox):
                return fragment
        return None

    def _build_ai_snapshot() -> dict:
        """Return the current AI-visible state for diagnostics."""
        enemies = [
            {
                "x": rec["x"],
                "y": rec["y"],
                "rank": rec["rank"],
                "health_points": rec["health_points"],
            }
            for rec in enemy_data.values()
            if rec["alive"]
        ]
        incoming = []
        for shot in enemy_shots:
            bbox = canvas.bbox(shot["item"])
            if not bbox:
                continue
            incoming.append(
                {
                    "x": (bbox[0] + bbox[2]) / 2,
                    "y": bbox[1],
                    "velocity_y": shot["velocity_y"],
                }
            )
        player_projectiles = [
            {
                "x": (
                    canvas.bbox(shot["item"])[0] + canvas.bbox(shot["item"])[2]
                )
                / 2
            }
            for shot in player_shots
            if canvas.bbox(shot["item"])
        ]
        general_record = enemy_data.get(general_id) if general_id else None
        general_x = general_record["x"] if general_record else player["x"]
        elapsed = min(_elapsed_seconds(), MAX_RUN_SECONDS)
        time_remaining_fraction = max(
            0.0, 1.0 - elapsed / MAX_RUN_SECONDS if MAX_RUN_SECONDS else 1.0
        )
        return {
            "player_x": player["x"],
            "player_y": player["y"],
            "canvas_width": canvas_width,
            "enemies": enemies,
            "incoming": incoming,
            "general_x": general_x,
            "charges": charge_count,
            "charge_capacity": charge_capacity,
            "player_shots": player_projectiles,
            "time_remaining_fraction": time_remaining_fraction,
            "elapsed_seconds": elapsed,
        }

    def _update_entities() -> None:
        """Advance every entity (shots, enemies, debris) each tick."""
        nonlocal charge_count, general_ai
        if paused:
            return
        _update_shooting_stars()
        if not game_over and _elapsed_seconds() >= MAX_RUN_SECONDS:
            action_var.set(
                "Time limit reached! Earth falls. Click Reset to restart."
            )
            _player_defeated()
            return
        player_box = canvas.bbox(player_item)

        for shot in list(player_shots):
            canvas.move(shot["item"], 0, shot["velocity_y"])
            coords = canvas.coords(shot["item"])
            if not coords or coords[1] < 0:
                canvas.delete(shot["item"])
                player_shots.remove(shot)
                continue
            hit = _check_enemy_collision(shot["item"])
            if hit:
                canvas.delete(shot["item"])
                player_shots.remove(shot)
                _destroy_enemy(hit)
                continue
            dart = _check_projectile_overlap(shot["item"], enemy_shots)
            if dart:
                canvas.delete(shot["item"])
                player_shots.remove(shot)
                canvas.delete(dart["item"])
                enemy_shots.remove(dart)
                continue
            fragment_hit = _check_debris_collision(shot["item"])
            if fragment_hit:
                canvas.delete(shot["item"])
                player_shots.remove(shot)
                canvas.delete(fragment_hit["item"])
                debris.remove(fragment_hit)
                continue

        for shot in list(enemy_shots):
            canvas.move(
                shot["item"], shot.get("velocity_x", 0.0), shot["velocity_y"]
            )
            coords = canvas.coords(shot["item"])
            if not coords or coords[3] > canvas_height:
                canvas.delete(shot["item"])
                enemy_shots.remove(shot)
                continue
            dart = _check_projectile_overlap(shot["item"], player_shots)
            if dart:
                canvas.delete(shot["item"])
                enemy_shots.remove(shot)
                canvas.delete(dart["item"])
                player_shots.remove(dart)
                continue
            if player_box:
                overlaps = canvas.find_overlapping(*player_box)
                if shot["item"] in overlaps:
                    canvas.delete(shot["item"])
                    enemy_shots.remove(shot)
                    _handle_player_hit()

        for bomb in list(bombs):
            canvas.move(bomb["item"], 0, bomb["velocity_y"])
            coords = canvas.coords(bomb["item"])
            if not coords or coords[1] < 0:
                canvas.delete(bomb["item"])
                bombs.remove(bomb)
                continue
            hit = _check_enemy_collision(bomb["item"])
            if hit:
                explosion = True
                canvas.delete(bomb["item"])
                bombs.remove(bomb)
                _destroy_enemy(hit, explosion=explosion)
                if explosion:
                    source = enemy_data.get(hit)
                    if source:
                        for eid, record in enemy_data.items():
                            if (
                                record["alive"]
                                and eid != hit
                                and abs(record["x"] - source["x"]) <= 80
                                and abs(record["y"] - source["y"]) <= 80
                            ):
                                _destroy_enemy(eid, explosion=True)

        for charge in list(charges):
            canvas.move(charge["item"], 0, charge["velocity_y"])
            coords = canvas.coords(charge["item"])
            if coords and coords[3] > canvas_height:
                canvas.delete(charge["item"])
                charges.remove(charge)
                continue
            charge_box = canvas.bbox(charge["item"])
            if (
                charge_box
                and player_box
                and charge_count < charge_capacity
                and charge_box[0] < player_box[2]
                and charge_box[2] > player_box[0]
                and charge_box[1] < player_box[3]
                and charge_box[3] > player_box[1]
            ):
                canvas.delete(charge["item"])
                charges.remove(charge)
                charge_count = min(charge_capacity, charge_count + 1)
                action_var.set(
                    "Space charge secured "
                    f"({charge_count}/{charge_capacity}). "
                    "Right-click to deploy."
                )
                _update_status()
                break

        for fragment in list(debris):
            fragment["velocity_y"] = min(fragment["velocity_y"] + 0.04, 5.0)
            fragment["velocity_x"] *= 0.99
            canvas.move(
                fragment["item"],
                fragment["velocity_x"],
                fragment["velocity_y"],
            )
            coords = canvas.coords(fragment["item"])
            if not coords or coords[1] > canvas_height - ground_height + 5:
                canvas.delete(fragment["item"])
                debris.remove(fragment)
                continue
            frag_box = canvas.bbox(fragment["item"])
            if (
                frag_box
                and player_box
                and frag_box[0] < player_box[2]
                and frag_box[2] > player_box[0]
                and frag_box[1] < player_box[3]
                and frag_box[3] > player_box[1]
            ):
                canvas.delete(fragment["item"])
                debris.remove(fragment)
                action_var.set("Debris hit your ship!")
                _handle_player_hit()
                continue
            if debris_damages_all:
                enemy_hit = _check_enemy_collision(fragment["item"])
                if enemy_hit:
                    canvas.delete(fragment["item"])
                    debris.remove(fragment)
                    _destroy_enemy(enemy_hit)
                    continue

        if general_id:
            record = enemy_data.get(general_id)
            if record and record["alive"]:
                rail_y = record.get("rail_y", record["y"])
                bbox = canvas.bbox(record["item"])
                if bbox:
                    current_center_y = (bbox[1] + bbox[3]) / 2
                    delta_y = rail_y - current_center_y
                    if abs(delta_y) > 1:
                        canvas.move(record["item"], 0, delta_y)
                record["y"] = rail_y
                safe_margin = 25
                if general_ai.get("target") is None:
                    general_ai["target"] = canvas_width - safe_margin
                general_ai["cooldown"] = max(
                    0.0, general_ai.get("cooldown", 0.0) - 1
                )
                gap = player["x"] - record["x"]
                player_lingering = _player_lingering()
                player_edge_distance = min(
                    player["x"], canvas_width - player["x"]
                )
                near_player_edge = player_edge_distance < 90
                near_player_edge = (
                    min(player["x"], canvas_width - player["x"]) < 90
                )
                if player_lingering:
                    general_ai["mode"] = random.choice(["pressure", "harass"])
                    offset = 0.0
                    if general_ai["mode"] == "harass":
                        offset = random.uniform(-90, 90)
                    general_ai["target"] = max(
                        safe_margin,
                        min(canvas_width - safe_margin, player["x"] + offset),
                    )
                    general_ai["cooldown"] = random.randint(15, 35)
                elif general_ai.get("mode") == "pressure":
                    general_ai["mode"] = "patrol"
                    general_ai["cooldown"] = random.randint(50, 110)

                if (
                    near_player_edge
                    and not general_ai.get("retreat", False)
                    and random.random() < 0.4
                ):
                    far_side = (
                        canvas_width - safe_margin
                        if player["x"] < canvas_width / 2
                        else safe_margin
                    )
                    general_ai["retreat"] = True
                    general_ai["mode"] = "retreat"
                    general_ai["retreat_target"] = far_side
                    general_ai["cooldown"] = random.randint(60, 110)

                if random.random() < 0.12 and general_ai.get("mode") not in (
                    "pressure",
                    "harass",
                ):
                    flank_direction = -1 if random.random() < 0.5 else 1
                    flank_offset = 160 + random.uniform(-110, 110)
                    general_ai["mode"] = "flank"
                    general_ai["target"] = max(
                        safe_margin,
                        min(
                            canvas_width - safe_margin,
                            player["x"] + flank_direction * flank_offset,
                        ),
                    )
                    general_ai["cooldown"] = random.randint(60, 120)

                if general_ai.get("mode") != "pressure":
                    if abs(gap) < 140:
                        general_ai["mode"] = "evade"
                        direction = -1 if gap > 0 else 1
                        general_ai["target"] = max(
                            safe_margin,
                            min(
                                canvas_width - safe_margin,
                                record["x"]
                                + direction * (200 + random.uniform(0, 80)),
                            ),
                        )
                        general_ai["cooldown"] = 35
                    elif abs(gap) > 220:
                        general_ai["mode"] = "stalk"
                        offset = random.uniform(-80, 80)
                        general_ai["target"] = max(
                            safe_margin,
                            min(
                                canvas_width - safe_margin,
                                player["x"] + offset,
                            ),
                        )
                        general_ai["cooldown"] = random.randint(40, 80)
                    elif general_ai["cooldown"] <= 0:
                        general_ai["mode"] = "patrol"
                        general_ai["target"] = random.choice(
                            [safe_margin, canvas_width - safe_margin]
                        )
                        general_ai["cooldown"] = random.randint(60, 140)
                target = general_ai.get("target", canvas_width / 2)
                if general_ai.get("retreat"):
                    target = general_ai.get("retreat_target", target)
                if (
                    abs(target - record["x"]) < 5
                    and general_ai["mode"] == "evade"
                ):
                    general_ai["mode"] = "patrol"
                    general_ai["cooldown"] = random.randint(40, 100)
                if (
                    general_ai.get("retreat")
                    and abs(target - record["x"]) < 12
                ):
                    general_ai["retreat"] = False
                    general_ai["retreat_target"] = None
                    general_ai["mode"] = "patrol"
                    general_ai["cooldown"] = random.randint(60, 120)
                direction_to_target = 0
                if target > record["x"]:
                    direction_to_target = 1
                elif target < record["x"]:
                    direction_to_target = -1
                mode = general_ai.get("mode")
                base_speed = 4
                if mode == "evade":
                    base_speed = 7
                elif mode == "pressure":
                    base_speed = 6.5
                elif mode == "harass":
                    base_speed = 6.0
                elif mode == "flank":
                    base_speed = 5.8
                elif mode == "stalk":
                    base_speed = 5.2
                speed = base_speed + random.uniform(-0.8, 0.8)
                speed = min(speed, general_speed_limit)
                movement = direction_to_target * speed
                if abs(target - record["x"]) < abs(movement):
                    movement = target - record["x"]
                jitter = random.uniform(-0.6, 0.6)
                general_vel = general_ai.get("velocity_x", 0.0)
                general_accel = max(0.25, min(1.0, general_speed_limit * 0.15))
                desired_velocity = movement
                general_vel += max(
                    -general_accel,
                    min(general_accel, desired_velocity - general_vel),
                )
                general_vel += random.uniform(-0.4, 0.4)
                general_ai["velocity_x"] = max(
                    -general_speed_limit, min(general_speed_limit, general_vel)
                )
                new_x = record["x"] + general_ai["velocity_x"] + jitter
                new_x = max(
                    safe_margin, min(canvas_width - safe_margin, new_x)
                )
                canvas.move(record["item"], new_x - record["x"], 0)
                record["x"] = new_x
                if (
                    general_ai.get("mode") in ("pressure", "harass")
                    and not player_lingering
                ):
                    general_ai["mode"] = "patrol"
                if (
                    general_ai.get("mode") == "flank"
                    and general_ai["cooldown"] <= 0
                ):
                    general_ai["mode"] = "stalk"

    def _tick() -> None:
        """Drive the per-frame simulation loop (moves, AI, entities)."""
        nonlocal tick_handle, player_last_x, player_idle_ticks
        if game_over:
            return
        if paused:
            tick_handle = _scaled_after(40, _tick)
            return
        if abs(player["x"] - player_last_x) < 2:
            player_idle_ticks = min(player_idle_ticks + 1, 10000)
        else:
            player_idle_ticks = 0
        player_last_x = player["x"]
        scale = _time_scale()
        steps = _tick_steps(scale)
        for _ in range(int(steps)):
            if game_over or paused:
                break
            _update_player_motion()
            _sample_edge_discipline()
            _update_entities()
        _update_status()
        tick_handle = _scaled_after(40, _tick)

    class AIPilotController:
        """Lightweight controller that bridges the AI brain and the canvas."""

        def __init__(self) -> None:
            """Prepare the autopilot controller for the Tk loop."""
            self.running = False
            self.handle: str | None = None

        def start(self) -> None:
            """Begin the AI pilot background loop."""
            nonlocal autopilot_active
            if self.running:
                return
            autopilot_active = True
            self.running = True
            _ensure_game_started()
            ai_button.configure(text="Stop AI helper")
            action_var.set("AI pilot engaged. Monitoring battlefield...")
            self._loop()

        def stop(self) -> None:
            """Stop the AI pilot and restore manual control."""
            nonlocal autopilot_active
            if not self.running:
                return
            self.running = False
            autopilot_active = False
            if self.handle:
                try:
                    canvas.after_cancel(self.handle)
                except Exception:  # pragma: no cover - Tk teardown
                    pass
                self.handle = None
            ai_button.configure(text="Let AI take care")

        def _loop(self) -> None:
            """Run the AI decision loop until stopped or the game ends."""
            if not self.running:
                return
            if game_over:
                if learning_mode:
                    self.handle = _scaled_after(80, self._loop)
                    return
                self.stop()
                return
            if paused:
                self.handle = _scaled_after(120, self._loop)
                return
            snapshot = _build_ai_snapshot()
            decision = ai_brain.decide(snapshot)
            move_delta = decision.get("move", 0.0)
            move_delta = max(-1.0, min(1.0, move_delta))
            if move_delta:
                _move_player_to(player["x"] + move_delta * player_speed_limit)
            if decision.get("shoot"):
                _fire_player_shot(None)
            if decision.get("charge"):
                _handle_right_click(None, announce=False)
            self.handle = _scaled_after(80, self._loop)

    ai_controller = AIPilotController()

    def _spawn_debris(
        x: float,
        y: float,
        *,
        count: int | None = None,
        speed_scale: float = 1.0,
    ) -> None:
        """Create some explosion debris particles around a point."""
        base_count = count if count is not None else debris_default_count
        actual_count = max(1, int(base_count * explosion_violence))
        for _ in range(actual_count):
            speed = random.uniform(1.0, 4.0) * speed_scale * explosion_violence
            horizontal_velocity = speed * random.uniform(-1.0, 1.0)
            vertical_velocity = -abs(speed) + random.uniform(-1.0, 2.0)
            debris_piece_id = canvas.create_polygon(
                x - 4,
                y - 4,
                x + 4,
                y - 4,
                x,
                y + 4,
                fill="#ffd1a3",
                outline="#ffae6a",
            )
            debris.append(
                {
                    "item": debris_piece_id,
                    "velocity_x": horizontal_velocity,
                    "velocity_y": vertical_velocity,
                }
            )

    def _start_timers() -> None:
        """Kick off the recurring timers for ticks, firing and charges."""
        nonlocal tick_handle, fire_handle, charge_handle, general_fire_handle
        tick_handle = _scaled_after(40, _tick)
        fire_handle = _scaled_after(900, _enemy_fire_cycle)
        charge_handle = _scaled_after(12000, _charge_cycle)
        general_fire_handle = _scaled_after(
            random.randint(600, 1200), _general_fire_cycle
        )

    def _accept_seed() -> None:
        """Apply the pending kill order as a seed and close the window."""
        if pending_order is None:
            return
        _apply_seed(pending_order, pending_duration)
        _close_window()

    accept_button.configure(command=_accept_seed)
    try_again_button.configure(command=_reset_game)
    cancel_button.configure(command=_close_window)

    canvas.bind("<Motion>", _move_player)
    canvas.bind("<Button-1>", _fire_player_shot)
    canvas.bind("<Button-3>", _handle_right_click)
    canvas.bind("<Button-2>", _handle_right_click)
    canvas.bind("<Control-Button-1>", _handle_right_click)
    window.protocol("WM_DELETE_WINDOW", _close_window)
    window.grab_set()
    _spawn_enemies()
    _update_status()
    action_var.set("Click anywhere in the field to begin.")
