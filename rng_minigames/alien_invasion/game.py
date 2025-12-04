"""Alien invasion mini-game for seed forging."""

from __future__ import annotations

import hashlib
import math
import random
import time
from pathlib import Path
from typing import TYPE_CHECKING, Callable

try:  # pragma: no cover - Tk only available with GUI rendering
    import tkinter as tk
    from tkinter import ttk
except Exception:  # pragma: no cover - executed when Tk is unavailable
    tk = None
    ttk = None

from .ai_agent import AlienInvasionAI
from .ai_config import load_settings as load_ai_settings
from .game_config import load_settings as load_game_settings
from .hall_of_fame import HallOfFame
from rng_minigames.api import MinigameContext


def launch_alien_invasion(context: MinigameContext) -> None:
    """Space-invader inspired mini-game."""

    def _apply_seed(order: list[str], duration: float) -> None:
        payload = "|".join(order) + f"|{int(duration * 1000)}"
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        seed_value = str(int(digest[:14], 16))
        context.set_seed(seed_value)
        context.notify(
            f"Alien invasion forged seed {seed_value}.",
            "INFO",
        )

    if not context.render or tk is None or context.tk_root is None:
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

    window = tk.Toplevel(context.tk_root)
    window.title("Alien invasion")
    window.resizable(False, False)
    window.transient(context.tk_root)
    canvas_width = int(760 * 1.5)
    canvas_height = int(480 * 1.5)
    canvas = tk.Canvas(
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
    learning_speed_var: "tk.IntVar | None" = None
    player_explosion: list[dict] = []
    player_explosion_handle: str | None = None
    player_explosion_end: float | None = None
    player_explosion_active = False
    player_auto_reset_handle: str | None = None

    def _clamp_learning_speed(value: int) -> int:
        return max(1, min(10, value))

    def _current_learning_speed() -> int:
        nonlocal learning_speed_multiplier
        value = learning_speed_multiplier
        if learning_speed_var is not None:
            try:
                value = int(learning_speed_var.get())
            except Exception:
                value = learning_speed_multiplier
        value = _clamp_learning_speed(value)
        learning_speed_multiplier = value
        return value

    def _handle_learning_speed_change(*_args: object) -> None:
        nonlocal learning_speed_multiplier
        if learning_speed_var is None:
            return
        current = learning_speed_var.get()
        try:
            value = int(current)
        except Exception:
            value = learning_speed_multiplier
        value = _clamp_learning_speed(value)
        learning_speed_multiplier = value
        if current != value:
            learning_speed_var.set(value)

    def _time_scale() -> float:
        return _current_learning_speed() if learning_mode else 1.0

    def _draw_background() -> None:
        gradient_steps = 60
        last_color = "#040912"
        for step in range(gradient_steps):
            ratio = step / gradient_steps
            r = int(5 + ratio * 20)
            g = int(6 + ratio * 25)
            b = int(12 + ratio * 35)
            color = f"#{r:02x}{g:02x}{b:02x}"
            last_color = color
            y0 = sky_height * (step / gradient_steps)
            y1 = sky_height * ((step + 1) / gradient_steps)
            canvas.create_rectangle(
                0, y0, canvas_width, y1, fill=color, outline=""
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
            if len(ridge_points) < 2:
                return sky_height
            for (x0, y0), (x1, y1) in zip(ridge_points[:-1], ridge_points[1:]):
                if (x0 <= x_val <= x1) or (x1 <= x_val <= x0):
                    if x1 == x0:
                        return (y0 + y1) / 2
                    ratio = (x_val - x0) / (x1 - x0)
                    return y0 + ratio * (y1 - y0)
            return ridge_points[-1][1]

        def _draw_skyline() -> None:
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
                        wx = left + column_offset - 1
                        wy = base_y - 4
                        while wy > top_y + 2:
                            lit = random.random() < 0.35
                            canvas.create_rectangle(
                                wx,
                                wy,
                                wx + 1,
                                wy + 1,
                                fill="#ffdd7a" if lit else "#4c5d89",
                                outline="",
                            )
                            wy -= 4
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
            if len(ridge_points) < 2:
                return
            cluster_count = random.randint(3, 6)
            centers = [
                (random.uniform(0, canvas_width), sky_height + random.uniform(-10, 8))
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
        return tuple(int(hex_color[i : i + 2], 16) for i in (1, 3, 5))

    def _color_for_star(base: tuple[int, int, int], brightness: float) -> str:
        brightness = max(0.0, min(1.0, brightness))
        r = int(base[0] * brightness + 255 * (1 - brightness))
        g = int(base[1] * brightness + 255 * (1 - brightness))
        b = int(base[2] * brightness + 255 * (1 - brightness))
        return f"#{r:02x}{g:02x}{b:02x}"

    def _clear_shooting_stars() -> None:
        for star in shooting_stars:
            canvas.delete(star["head"])
            canvas.delete(star["tail"])
        shooting_stars.clear()

    def _schedule_next_shooting_star(multiplier: float = 1.0) -> None:
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
        vx = math.cos(angle) * speed
        vy = math.sin(angle) * speed
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
                "vx": vx,
                "vy": vy,
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
        nonlocal shooting_stars
        if learning_mode:
            return
        now = time.time()
        while now >= next_shooting_star_time:
            _spawn_shooting_star()
            now = time.time()
        for star in list(shooting_stars):
            speed = math.hypot(star["vx"], star["vy"]) or 1.0
            ux = star["vx"] / speed
            uy = star["vy"] / speed
            star["head_x"] += star["vx"]
            star["head_y"] += star["vy"]
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
            tail_x = star["head_x"] - ux * star["current_len"]
            tail_y = star["head_y"] - uy * star["current_len"]
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
    instructions = ttk.Label(
        window,
        text=(
            "Move with the mouse, left-click to fire, right-click (or ctrl-click) "
            "to launch stored space charges. Catch capsules to stockpile up to "
            "three and watch the shields, charges and countdown timers."
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
    general_speed_limit = float(general_cfg.get("max_speed", 7.0))
    player_shield_max = max(1, int(player_cfg.get("shield", 50)))
    player_speed_limit = float(
        motion_cfg.get(
            "max_speed",
            general_speed_limit * 2 if general_speed_limit > 0 else 14.0,
        )
    )
    player_speed_limit = max(player_speed_limit, 1.0)
    player_accel = max(0.05, float(motion_cfg.get("accel", 0.45)))
    player_decel = max(0.05, float(motion_cfg.get("decel", 0.4)))
    motion_snap_error = max(0.1, float(motion_cfg.get("snap_error", 1.2)))
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
    status_frame = ttk.Frame(window)
    status_frame.pack(fill="x", padx=16, pady=(0, 2))
    shield_status_var = tk.StringVar()
    ttk.Label(
        status_frame,
        textvariable=shield_status_var,
        font=("Helvetica", 15, "bold"),
    ).pack(anchor="w", pady=(0, 2))
    action_var = tk.StringVar()
    ttk.Label(status_frame, textvariable=action_var).pack(
        anchor="w", pady=(0, 4)
    )
    ai_stats_var = tk.StringVar()

    def _update_ai_stats() -> None:
        runs = ai_brain.state.get("runs", 0)
        saved = ai_brain.state.get("worlds_saved", 0)
        lost = ai_brain.state.get("worlds_lost", 0)
        ai_stats_var.set(
            "AI games: {runs}     Everybody lives: {saved}     Everybody dies: {lost}".format(
                runs=runs, saved=saved, lost=lost
            )
        )

    _update_ai_stats()
    ttk.Label(
        status_frame,
        textvariable=ai_stats_var,
        font=("Helvetica", 10, "normal"),
    ).pack(anchor="w", pady=(0, 8))
    button_bar = ttk.Frame(window)
    button_bar.pack(fill="x", padx=16, pady=(2, 12))
    accept_button = ttk.Button(button_bar, text="Use seed", state=tk.DISABLED)
    accept_button.pack(side="right", padx=(0, 8))
    cancel_button = ttk.Button(button_bar, text="Cancel")
    cancel_button.pack(side="right", padx=(0, 8))
    try_again_button = ttk.Button(button_bar, text="Reset")
    try_again_button.pack(side="right", padx=(0, 8))

    pause_button = ttk.Button(button_bar, text="Pause")
    pause_button.pack(side="left", padx=(0, 8))

    def _show_hall_of_fame() -> None:
        hall_of_fame.show(window)

    def _ai_in_control() -> bool:
        return autopilot_active or learning_mode

    def _reward_enemy_destroyed(record: dict) -> None:
        if not _ai_in_control():
            return
        rank = record.get("rank", "lieutenant")
        ai_brain.reward_enemy_destroyed(rank, general=record.get("general", False))

    def _penalize_enemy_respawned(record: dict) -> None:
        if not _ai_in_control():
            return
        rank = record.get("rank", "lieutenant")
        ai_brain.penalize_enemy_respawned(rank)

    def _cancel_player_auto_reset() -> None:
        nonlocal player_auto_reset_handle
        if player_auto_reset_handle:
            try:
                canvas.after_cancel(player_auto_reset_handle)
            except Exception:
                pass
            player_auto_reset_handle = None

    def _schedule_player_auto_reset() -> None:
        nonlocal player_auto_reset_handle
        if learning_mode:
            return
        _cancel_player_auto_reset()
        delay_ms = max(100, int(player_explosion_hold * 1000))

        def _auto_reset() -> None:
            _reset_game()

        player_auto_reset_handle = canvas.after(delay_ms, _auto_reset)

    def _clear_player_explosion() -> None:
        nonlocal player_explosion_handle, player_explosion_end, player_explosion_active
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
        nonlocal player_explosion_handle, player_explosion_active
        if not player_explosion or player_explosion_end is None:
            _clear_player_explosion()
            return
        now = time.time()
        for shard in list(player_explosion):
            canvas.move(shard["item"], shard["vx"], shard["vy"])
            shard["vy"] += 0.08 * explosion_violence
            shard["life"] -= 0.015
            coords = canvas.coords(shard["item"])
            if not coords:
                player_explosion.remove(shard)
                continue
            cx = (coords[0] + coords[2]) / 2
            cy = (coords[1] + coords[3]) / 2
            size = max(0.5, shard["size"] * shard["life"])
            canvas.coords(
                shard["item"],
                cx - size,
                cy - size,
                cx + size,
                cy + size,
            )
            if shard["life"] <= 0:
                canvas.delete(shard["item"])
                player_explosion.remove(shard)
        if now >= player_explosion_end or not player_explosion:
            _clear_player_explosion()
            return
        player_explosion_handle = canvas.after(
            explosion_frame_ms, _animate_player_explosion
        )

    def _start_player_explosion() -> None:
        nonlocal player_explosion_end, player_explosion_active
        _clear_player_explosion()
        player_explosion_end = time.time() + explosion_duration
        player_explosion_active = True
        canvas.itemconfigure(player_item, state="hidden")
        colors = ["#ffd166", "#ff8a5b", "#ff4d6d", "#ffe29a", "#ffb347"]
        size_scale = max(0.5, min(explosion_violence, 2.0))
        for _ in range(explosion_shard_count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(2.5, 6.5) * explosion_violence
            size = random.uniform(3.5, 8.5) * size_scale
            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed
            color = random.choice(colors)
            item = canvas.create_oval(
                player["x"] - size,
                player["y"] - size,
                player["x"] + size,
                player["y"] + size,
                fill=color,
                outline="",
            )
            player_explosion.append(
                {
                    "item": item,
                    "vx": vx,
                    "vy": vy,
                    "size": size,
                    "life": random.uniform(0.8, 1.2),
                }
            )
        player_explosion_handle = canvas.after(
            explosion_frame_ms, _animate_player_explosion
        )

    hall_button = ttk.Button(
        button_bar, text="Hall of fame", command=_show_hall_of_fame
    )
    hall_button.pack(side="left", padx=(0, 8))

    def _toggle_ai_pilot() -> None:
        if ai_controller.running:
            ai_controller.stop()
            action_var.set("AI disengaged. Manual control restored.")
        else:
            ai_controller.start()

    ai_button = ttk.Button(
        button_bar, text="Let AI take care", command=_toggle_ai_pilot
    )
    ai_button.pack(side="left", padx=(0, 8))

    def _toggle_learning() -> None:
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

    learning_button = ttk.Button(
        button_bar, text="Let AI learn", command=_toggle_learning
    )
    learning_button.pack(side="left", padx=(0, 8))

    if tk is not None and ttk is not None:
        learning_speed_var = tk.IntVar(value=learning_speed_multiplier)
        learning_speed_var.trace_add("write", _handle_learning_speed_change)
        ttk.Label(button_bar, text="Learning speed").pack(
            side="left", padx=(8, 4)
        )
        learning_speed_spin = ttk.Spinbox(
            button_bar,
            from_=1,
            to=10,
            width=4,
            textvariable=learning_speed_var,
            justify="center",
            command=_handle_learning_speed_change,
        )
        learning_speed_spin.pack(side="left", padx=(0, 8))
    else:
        learning_speed_var = None

    def _perform_ai_forget() -> None:
        ai_brain.forget()
        _update_ai_stats()
        action_var.set("AI memory wiped. Fresh slate!")

    def _request_forget() -> None:
        if tk is None:
            _perform_ai_forget()
            return
        dialog = tk.Toplevel(window)
        dialog.title("Let AI forget?")
        dialog.resizable(False, False)
        ttk.Label(
            dialog,
            text="Are you sure you will wipe the poor fella's memory?",
            wraplength=320,
            padding=12,
        ).pack(fill="x")
        btn_row = ttk.Frame(dialog)
        btn_row.pack(padx=12, pady=(0, 12))

        def _wipe() -> None:
            _perform_ai_forget()
            dialog.destroy()

        ttk.Button(btn_row, text="Wipe", command=_wipe).pack(
            side="left", padx=(0, 8)
        )
        ttk.Button(btn_row, text="Pardon", command=dialog.destroy).pack(
            side="left", padx=(0, 8)
        )
        dialog.transient(window)
        dialog.grab_set()
        dialog.protocol("WM_DELETE_WINDOW", dialog.destroy)

    forget_button = ttk.Button(
        button_bar, text="Let AI forget", command=_request_forget
    )
    forget_button.pack(side="left", padx=(0, 8))

    def _toggle_pause() -> None:
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
    player_width = 26
    player_height = 22
    def _player_shape_coords(cx: float, cy: float) -> list[float]:
        return [
            cx,
            cy - 22,
            cx + 10,
            cy - 10,
            cx + 14,
            cy - 2,
            cx + 10,
            cy + 8,
            cx + 4,
            cy + 16,
            cx - 4,
            cy + 16,
            cx - 10,
            cy + 8,
            cx - 14,
            cy - 2,
            cx - 10,
            cy - 10,
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
        step = player_accel if abs(desired_velocity) > abs(player_velocity) else player_decel
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
    general_ai = {"target": None, "mode": "patrol", "cooldown": 0.0, "vx": 0.0}
    charge_count = 0
    destroyed_stack: list[str] = []
    pending_order: list[str] | None = None
    pending_duration = 0.0
    game_over = False
    last_shooter: str | None = None
    general_hits = 0
    player_hits = 0
    tick_handle: str | None = None
    fire_handle: str | None = None
    charge_handle: str | None = None
    general_fire_handle: str | None = None

    def _update_enemy_shield_visual(enemy_id: str) -> None:
        record = enemy_data.get(enemy_id)
        if not record:
            return
        rank = record.get("rank", "lieutenant")
        outline = ""
        width = 0
        if rank == "colonel":
            ratio = record["hp"] / record["max_hp"]
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
            width = 2 if record["hp"] > 1 else 0
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
                    not general_group
                    and row_index == 1
                    and cols >= 2
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
                for c in range(row_cols):
                    general = general_group
                    x = start_x + c * spacing
                    if (
                        stagger_row
                        and row_cols > 1
                        and c == row_cols - 1
                        and not general
                    ):
                        continue
                    eid += 1
                    if not general:
                        x = max(field_margin, min(canvas_width - field_margin, x))
                        row_positions.append(x)
                    if general:
                        rank = "general"
                        max_hp = general_shield_max
                    elif is_bottom:
                        rank = "colonel"
                        max_hp = 5
                    elif is_major:
                        rank = "major"
                        max_hp = 2
                    else:
                        rank = "lieutenant"
                        max_hp = 1
                    hp = max_hp
                    item = _enemy_polygon(
                        x, y, general=general, elite=is_bottom or is_major
                    )
                    enemy_id = f"E{eid:02d}"
                    enemy_data[enemy_id] = {
                        "item": item,
                        "x": x,
                        "y": y,
                        "spawn_x": x,
                        "spawn_y": y,
                        "alive": True,
                        "general": general,
                        "hp": hp,
                        "max_hp": max_hp,
                        "rank": rank,
                        "rail_y": y if general else None,
                    }
                    if general:
                        general_id = enemy_id
                        general_ai = {
                            "target": canvas_width - field_margin,
                            "mode": "patrol",
                            "cooldown": random.randint(60, 120),
                            "vx": 0.0,
                        }
                    _update_enemy_shield_visual(enemy_id)
                if (
                    stagger_row
                    and not general_group
                    and row_positions
                ):
                    if len(row_positions) >= 2:
                        delta = row_positions[1] - row_positions[0]
                    else:
                        delta = spacing or (canvas_width - 2 * field_margin) / max(
                            cols - 1, 1
                        )
                    delta = max(delta, 20)
                    extensions = [
                        row_positions[0] - delta,
                        row_positions[-1] + delta,
                    ]
                    for offset in (-1, 1):
                        extra_x = extensions[0] if offset < 0 else extensions[1]
                        extra_x = max(10, min(canvas_width - 10, extra_x))
                        eid += 1
                        x = extra_x
                        if is_bottom:
                            rank = "colonel"
                            max_hp = 5
                        elif is_major:
                            rank = "major"
                            max_hp = 2
                        else:
                            rank = "lieutenant"
                            max_hp = 1
                        hp = max_hp
                        item = _enemy_polygon(
                            x, y, general=False, elite=is_bottom or is_major
                        )
                        enemy_id = f"E{eid:02d}"
                        enemy_data[enemy_id] = {
                            "item": item,
                            "x": x,
                            "y": y,
                            "spawn_x": x,
                            "spawn_y": y,
                            "alive": True,
                            "general": False,
                            "hp": hp,
                            "max_hp": max_hp,
                            "rank": rank,
                            "rail_y": None,
                        }
                        _update_enemy_shield_visual(enemy_id)
        total_enemies = len(enemy_data)

    def _scaled_after(delay_ms: int, callback: Callable[[], None]) -> str:
        scale = _time_scale()
        scaled = max(1, int(delay_ms / scale))
        return canvas.after(scaled, callback)

    MAX_RUN_SECONDS = max_run_seconds

    def _elapsed_seconds() -> int:
        if run_start_time is None:
            return 0
        return max(0, int((time.time() - run_start_time) * _time_scale()))

    def _time_left_seconds() -> int:
        return max(0, MAX_RUN_SECONDS - _elapsed_seconds())

    def _format_time_left() -> str:
        remaining = _time_left_seconds()
        minutes, seconds = divmod(remaining, 60)
        return f"{minutes:02d}:{seconds:02d}"

    def _update_status() -> None:
        elapsed = _elapsed_seconds()
        player_remaining = max(player_shield_max - player_hits, 0)
        general_remaining = max(general_shield_max - general_hits, 0)
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
            f"General shield: {general_remaining}/{general_shield_max}"
            f"{gap_secondary}Neutron charges: {charge_count}/{charge_capacity}"
            f"{gap_timer}Time left: {_format_time_left()}{urgency}"
        )

    def _record_ai_outcome(
        success: bool, duration: float, *, controlling: bool | None = None
    ) -> None:
        if controlling is None:
            controlling = _ai_in_control()
        if not controlling:
            return
        ai_brain.record_run(success=success, duration=duration)
        _update_ai_stats()

    def _cancel_timers() -> None:
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
        nonlocal learning_restart_handle
        if learning_restart_handle:
            try:
                canvas.after_cancel(learning_restart_handle)
            except Exception:
                pass
            learning_restart_handle = None

    def _center_player() -> None:
        nonlocal player_last_x, player_idle_ticks
        player["y"] = canvas_height - ground_height - 60
        _set_player_target(canvas_width / 2, snap=True)
        player_last_x = player["x"]
        player_idle_ticks = 0

    def _player_lingering() -> bool:
        return player_idle_ticks >= PLAYER_LINGER_THRESHOLD

    def _reset_game(*, preserve_ai: bool = False) -> None:
        nonlocal run_start_time, kill_order, charge_count
        nonlocal general_hits, general_ai, timers_started
        nonlocal player_shots, enemy_shots, charges, bombs, destroyed_stack
        nonlocal pending_order, pending_duration, game_over, debris
        nonlocal player_hits, autopilot_active, ai_controller, completed_by_ai
        nonlocal paused, learning_mode, player_velocity, player_target_x
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
        general_ai = {"target": None, "mode": "patrol", "cooldown": 0.0, "vx": 0.0}
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
        nonlocal learning_restart_handle
        if not learning_mode:
            return
        _cancel_learning_restart()
        action_var.set("AI reviewing the battle log. Next sortie incoming...")
        def _restart() -> None:
            _reset_game(preserve_ai=True)
        learning_restart_handle = _scaled_after(1200, _restart)

    def _respawn_enemy(enemy_id: str, *, opposite: bool = False) -> None:
        record = enemy_data.get(enemy_id)
        if not record or record["alive"]:
            return
        spawn_x = record.get("spawn_x", record.get("x", canvas_width / 2))
        spawn_y = record.get("spawn_y", record.get("y", 80))
        record["x"] = spawn_x
        record["y"] = spawn_y
        record["hp"] = record["max_hp"]
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
        if not destroyed_stack:
            return None
        priorities = ("lieutenant", "major", "colonel", "general")
        for rank in priorities:
            for enemy_id in reversed(destroyed_stack):
                record = enemy_data.get(enemy_id)
                if record and record.get("rank") == rank and not record["alive"]:
                    destroyed_stack.remove(enemy_id)
                    return enemy_id
        return destroyed_stack.pop()

    def _player_defeated() -> None:
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
        _record_ai_outcome(success=False, duration=duration, controlling=controlling)
        if learning_mode:
            action_var.set("Training run failed. Relaunching immediately.")
            _reset_game(preserve_ai=True)
            return
        _start_player_explosion()
        action_var.set("Your shield collapsed! Reset or wait for auto-restart.")
        autopilot_active = False
        if ai_controller:
            ai_controller.stop()
        _schedule_player_auto_reset()

    def _handle_player_hit() -> None:
        nonlocal player_hits
        if game_over:
            return
        player_hits = min(player_shield_max, player_hits + 1)
        _update_status()
        if player_hits >= player_shield_max:
            action_var.set("Your shield collapsed! Reset now or wait a moment.")
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
        nonlocal pending_order, pending_duration, game_over, general_hits
        nonlocal completed_by_ai
        record = enemy_data.get(enemy_id)
        if not record or not record["alive"]:
            return
        if record["general"]:
            remaining = any(
                rec["alive"] and not rec["general"]
                for rec in enemy_data.values()
            )
            if remaining and general_hits < general_shield_max:
                general_hits += 1
                action_var.set(
                    "The general deflects the blast "
                    f"({general_hits}/{general_shield_max})."
                )
                _update_status()
                return
        elif record.get("hp", 1) > 1 and not explosion:
            record["hp"] -= 1
            remaining_hp = record["hp"]
            _update_enemy_shield_visual(enemy_id)
            descriptor = (
                "Colonel"
                if record.get("rank") == "colonel"
                else "Major"
                if record.get("rank") == "major"
                else "Cruiser"
            )
            action_var.set(
                f"{descriptor} absorbed the hit ({remaining_hp}/"
                f"{record['max_hp']} shields)."
            )
            return
        canvas.delete(record["item"])
        record["alive"] = False
        _reward_enemy_destroyed(record)
        destroyed_stack.append(enemy_id)
        kill_order.append(enemy_id)
        if explosion:
            action_var.set(
                "Space charge unleashed a chain reaction! Nice shot."
            )
            _spawn_debris(record["x"], record["y"], speed_scale=1.6)
        if record["general"]:
            general_hits = general_shield_max
        _update_status()
        if len(kill_order) >= total_enemies and total_enemies > 0:
            pending_order = kill_order[:]
            if run_start_time is not None:
                pending_duration = (time.time() - run_start_time) * _time_scale()
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

    def _fire_player_shot(_event: "tk.Event | None") -> None:
        nonlocal last_shot_time
        if len(player_shots) > 4 or game_over or paused:
            return
        now = time.perf_counter()
        if now - last_shot_time < 0.1:
            return
        _ensure_game_started()
        last_shot_time = now
        item = canvas.create_rectangle(
            player["x"] - 2,
            player["y"] - player_height / 2,
            player["x"] + 2,
            player["y"] - player_height / 2 - 12,
            fill="#ffffff",
            outline="",
        )
        player_shots.append({"item": item, "vy": -12})

    def _ensure_game_started() -> None:
        nonlocal timers_started, run_start_time
        if timers_started:
            return
        timers_started = True
        run_start_time = time.time()
        _start_timers()

    def _move_player_to(target_x: float, *, snap: bool = False) -> None:
        if game_over:
            return
        _set_player_target(target_x, snap=snap)

    def _move_player(event: "tk.Event") -> None:
        if game_over or paused:
            return
        _set_player_target(event.x)

    def _fire_enemy_shot(enemy_id: str, *, aim_for: float | None = None) -> bool:
        record = enemy_data.get(enemy_id)
        if not record or not record["alive"]:
            return False
        vy = 6
        vx = 0.0
        if record.get("rank") == "general":
            vy = 5
        item = canvas.create_rectangle(
            record["x"] - 4,
            record["y"] + 10,
            record["x"] + 4,
            record["y"] + 22,
            fill="#ff3366",
            outline="",
        )
        enemy_shots.append(
            {"item": item, "vy": vy, "vx": vx, "owner": enemy_id}
        )
        return True

    def _fire_weight(enemy_id: str) -> float:
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
        if len(charges) >= 2 or charge_count >= charge_capacity:
            return
        x = random.randint(60, canvas_width - 60)
        item = canvas.create_oval(
            x - 8, 30, x + 8, 46, fill="#b0f3ff", outline="#68d4ff"
        )
        charges.append({"item": item, "vy": 1.5})

    def _enemy_fire_cycle() -> None:
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
        nonlocal charge_handle
        if game_over:
            return
        if paused:
            charge_handle = _scaled_after(1000, _charge_cycle)
            return
        _spawn_charge_once()
        charge_handle = _scaled_after(6000, _charge_cycle)

    def _general_fire_cycle() -> None:
        nonlocal general_fire_handle
        if game_over:
            return
        if paused:
            general_fire_handle = _scaled_after(600, _general_fire_cycle)
            return
        if general_id:
            bursts = random.randint(1, 4)
            for _ in range(bursts):
                _fire_enemy_shot(general_id, aim_for=player["x"])
        interval = random.randint(600, 1200)
        general_fire_handle = _scaled_after(interval, _general_fire_cycle)

    def _launch_bomb() -> None:
        item = canvas.create_polygon(
            player["x"] - 6,
            player["y"] - 10,
            player["x"],
            player["y"] - 30,
            player["x"] + 6,
            player["y"] - 10,
            fill="#7dd9ff",
            outline="#bdefff",
        )
        bombs.append({"item": item, "vy": -1.5})

    def _handle_right_click(event: "tk.Event | None", *, announce: bool = True) -> None:
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
        bbox = canvas.bbox(item_id)
        if not bbox:
            return None
        overlaps = canvas.find_overlapping(*bbox)
        for eid, record in enemy_data.items():
            if record["alive"] and record["item"] in overlaps:
                return eid
        return None

    def _rects_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
        return not (a[2] <= b[0] or a[0] >= b[2] or a[3] <= b[1] or a[1] >= b[3])

    def _check_projectile_overlap(item_id: int, projectiles: list[dict]) -> dict | None:
        bbox = canvas.bbox(item_id)
        if not bbox:
            return None
        for projectile in projectiles:
            other_bbox = canvas.bbox(projectile["item"])
            if other_bbox and _rects_overlap(bbox, other_bbox):
                return projectile
        return None

    def _check_debris_collision(item_id: int) -> dict | None:
        bbox = canvas.bbox(item_id)
        if not bbox:
            return None
        for fragment in debris:
            other_bbox = canvas.bbox(fragment["item"])
            if other_bbox and _rects_overlap(bbox, other_bbox):
                return fragment
        return None

    def _build_ai_snapshot() -> dict:
        enemies = [
            {
                "x": rec["x"],
                "y": rec["y"],
                "rank": rec["rank"],
                "hp": rec["hp"],
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
                    "vy": shot["vy"],
                }
            )
        player_projectiles = [
            {"x": (canvas.bbox(shot["item"])[0] + canvas.bbox(shot["item"])[2]) / 2}
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
            canvas.move(shot["item"], 0, shot["vy"])
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
            canvas.move(shot["item"], shot.get("vx", 0.0), shot["vy"])
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
            canvas.move(bomb["item"], 0, bomb["vy"])
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
            canvas.move(charge["item"], 0, charge["vy"])
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
            fragment["vy"] = min(fragment["vy"] + 0.04, 5.0)
            fragment["vx"] *= 0.99
            canvas.move(fragment["item"], fragment["vx"], fragment["vy"])
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
                    random.random() < 0.12
                    and general_ai.get("mode") not in ("pressure", "harass")
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
                                record["x"] + direction * (200 + random.uniform(0, 80)),
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
                if abs(target - record["x"]) < 5 and general_ai["mode"] == "evade":
                    general_ai["mode"] = "patrol"
                    general_ai["cooldown"] = random.randint(40, 100)
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
                general_vel = general_ai.get("vx", 0.0)
                general_accel = max(
                    0.25, min(1.0, general_speed_limit * 0.15)
                )
                desired_velocity = movement
                general_vel += max(
                    -general_accel, min(general_accel, desired_velocity - general_vel)
                )
                general_vel += random.uniform(-0.4, 0.4)
                general_ai["vx"] = max(
                    -general_speed_limit, min(general_speed_limit, general_vel)
                )
                new_x = record["x"] + general_ai["vx"] + jitter
                new_x = max(safe_margin, min(canvas_width - safe_margin, new_x))
                canvas.move(record["item"], new_x - record["x"], 0)
                record["x"] = new_x
                if general_ai.get("mode") in ("pressure", "harass") and not player_lingering:
                    general_ai["mode"] = "patrol"
                if general_ai.get("mode") == "flank" and general_ai["cooldown"] <= 0:
                    general_ai["mode"] = "stalk"

    def _tick() -> None:
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
        steps = _current_learning_speed() if learning_mode else 1
        for _ in range(int(steps)):
            if game_over or paused:
                break
            _update_player_motion()
            _update_entities()
        _update_status()
        tick_handle = _scaled_after(40, _tick)

    class AIPilotController:
        """Lightweight controller that bridges the AI brain and the canvas."""

        def __init__(self) -> None:
            self.running = False
            self.handle: str | None = None

        def start(self) -> None:
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
        x: float, y: float, *, count: int | None = None, speed_scale: float = 1.0
    ) -> None:
        base_count = count if count is not None else debris_default_count
        actual_count = max(1, int(base_count * explosion_violence))
        for _ in range(actual_count):
            speed = random.uniform(1.0, 4.0) * speed_scale * explosion_violence
            vx = speed * random.uniform(-1.0, 1.0)
            vy = -abs(speed) + random.uniform(-1.0, 2.0)
            item = canvas.create_polygon(
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
                    "item": item,
                    "vx": vx,
                    "vy": vy,
                }
            )

    def _start_timers() -> None:
        nonlocal tick_handle, fire_handle, charge_handle, general_fire_handle
        tick_handle = _scaled_after(40, _tick)
        fire_handle = _scaled_after(900, _enemy_fire_cycle)
        charge_handle = _scaled_after(12000, _charge_cycle)
        general_fire_handle = _scaled_after(
            random.randint(600, 1200), _general_fire_cycle
        )

    def _accept_seed() -> None:
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
