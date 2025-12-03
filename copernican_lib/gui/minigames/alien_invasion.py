"""Alien invasion mini-game for seed forging."""

from __future__ import annotations

import hashlib
import random
import time
from typing import TYPE_CHECKING

try:  # pragma: no cover - Tk only available with GUI rendering
    import tkinter as tk
    from tkinter import ttk
except Exception:  # pragma: no cover - executed when Tk is unavailable
    tk = None
    ttk = None

if TYPE_CHECKING:  # pragma: no cover - typing only
    from copernican_lib.gui.app import CopernicanGUI


def launch_alien_invasion(
    host: "CopernicanGUI", seed_var: "tk.StringVar"
) -> None:
    """Space-invader inspired mini-game."""

    def _apply_seed(order: list[str], duration: float) -> None:
        payload = "|".join(order) + f"|{int(duration * 1000)}"
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        seed_value = str(int(digest[:14], 16))
        seed_var.set(seed_value)
        host.create_toast(
            f"Alien invasion forged seed {seed_value}.",
            severity="INFO",
            context="seed",
        )

    if not host.render or tk is None or host.root is None:
        dummy_order = [f"E{i}" for i in range(10)]
        random.shuffle(dummy_order)
        _apply_seed(dummy_order, random.random() * 20)
        return

    window = tk.Toplevel(host.root)
    window.title("Alien invasion")
    window.resizable(False, False)
    window.transient(host.root)
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
    instructions = ttk.Label(
        window,
        text=(
            "Move your ship with the mouse, left-click to fire, "
            "right-click to launch stored space charges. Catch capsules by "
            "touching them and destroy every invader to forge a seed."
        ),
        wraplength=canvas_width,
        padding=(12, 6),
    )
    instructions.pack(anchor="w", padx=16)
    status_var = tk.StringVar(value="Kills: 0 | Charge: empty")
    ttk.Label(window, textvariable=status_var, padding=(16, 2)).pack(
        anchor="w"
    )
    action_var = tk.StringVar()
    ttk.Label(window, textvariable=action_var, padding=(16, 0)).pack(
        anchor="w"
    )
    button_bar = ttk.Frame(window)
    button_bar.pack(fill="x", padx=16, pady=(8, 12))
    accept_button = ttk.Button(button_bar, text="Use seed", state=tk.DISABLED)
    accept_button.pack(side="right", padx=(0, 8))
    cancel_button = ttk.Button(button_bar, text="Cancel")
    cancel_button.pack(side="right", padx=(0, 8))
    try_again_button = ttk.Button(button_bar, text="Try again")
    try_again_button.pack(side="right", padx=(0, 8))

    start_time = time.time()
    kill_order: list[str] = []
    rows_config = [(4, 16), (1, 8), (1, 1)]
    total_enemies = sum(rows * cols for rows, cols in rows_config)
    player = {"x": canvas_width / 2, "y": canvas_height - 50}
    player_item = canvas.create_polygon(
        player["x"] - 20,
        player["y"] + 20,
        player["x"],
        player["y"] - 20,
        player["x"] + 20,
        player["y"] + 20,
        fill="#4df0ff",
        outline="#ffffff",
    )
    player_shots: list[dict] = []
    enemy_shots: list[dict] = []
    charges: list[dict] = []
    bombs: list[dict] = []
    enemy_data: dict[str, dict] = {}
    general_id: str | None = None
    general_direction = 1
    inventory_charge = False
    destroyed_stack: list[str] = []
    pending_order: list[str] | None = None
    pending_duration = 0.0
    game_over = False
    last_shooter: str | None = None
    tick_handle: str | None = None
    fire_handle: str | None = None
    charge_handle: str | None = None
    general_fire_handle: str | None = None

    def _enemy_polygon(x: float, y: float, general: bool = False) -> int:
        if general:
            return canvas.create_polygon(
                x - 30,
                y + 15,
                x,
                y - 30,
                x + 30,
                y + 15,
                x + 15,
                y + 25,
                x - 15,
                y + 25,
                fill="#ff679d",
                outline="#ffdbe8",
            )
        return canvas.create_polygon(
            x - 20,
            y + 10,
            x - 10,
            y - 10,
            x + 10,
            y - 10,
            x + 20,
            y + 10,
            x,
            y + 18,
            fill="#ffb347",
            outline="#ffe0b3",
        )

    def _spawn_enemies() -> None:
        nonlocal general_id
        general_id = None
        margin_x = 80
        row_gap = 70
        start_y = 80
        eid = 0
        row_counter = 0
        total_groups = len(rows_config)
        for group_index, (rows, cols) in enumerate(rows_config):
            general_group = group_index == total_groups - 1 and cols == 1
            for _ in range(rows):
                if cols > 1:
                    spacing = (canvas_width - 2 * margin_x) / max(cols - 1, 1)
                    total_span = spacing * (cols - 1)
                    start_x = (canvas_width - total_span) / 2
                else:
                    spacing = 0
                    start_x = canvas_width / 2
                y = start_y + row_counter * row_gap
                row_counter += 1
                for c in range(cols):
                    eid += 1
                    x = start_x + c * spacing
                    general = general_group
                    item = _enemy_polygon(x, y, general=general)
                    enemy_id = f"E{eid:02d}"
                    enemy_data[enemy_id] = {
                        "item": item,
                        "x": x,
                        "y": y,
                        "alive": True,
                        "general": general,
                    }
                    if general:
                        general_id = enemy_id

    def _update_status() -> None:
        status_var.set(
            f"Kills: {len(kill_order)}/{total_enemies} | Charge: "
            f"{'ready' if inventory_charge else 'empty'}"
        )

    def _cancel_timers() -> None:
        nonlocal tick_handle, fire_handle, charge_handle, general_fire_handle
        for handle in (tick_handle, fire_handle, charge_handle, general_fire_handle):
            if handle:
                try:
                    canvas.after_cancel(handle)
                except Exception:
                    pass
        tick_handle = fire_handle = charge_handle = general_fire_handle = None

    def _close_window() -> None:
        _cancel_timers()
        window.destroy()

    def _reset_game() -> None:
        nonlocal start_time, kill_order, inventory_charge
        nonlocal general_direction
        nonlocal player_shots, enemy_shots, charges, bombs, destroyed_stack
        nonlocal pending_order, pending_duration, game_over
        _cancel_timers()
        for record in enemy_data.values():
            canvas.delete(record["item"])
        enemy_data.clear()
        for projectile in player_shots + enemy_shots + bombs:
            canvas.delete(projectile["item"])
        for charge in charges:
            canvas.delete(charge["item"])
        player_shots.clear()
        enemy_shots.clear()
        bombs.clear()
        charges.clear()
        destroyed_stack.clear()
        kill_order = []
        general_direction = 1
        inventory_charge = False
        pending_order = None
        pending_duration = 0.0
        game_over = False
        start_time = time.time()
        _spawn_enemies()
        _update_status()
        action_var.set("")
        accept_button.state(["disabled"])
        _start_timers()

    def _respawn_enemy(enemy_id: str) -> None:
        record = enemy_data.get(enemy_id)
        if not record or record["alive"]:
            return
        record["item"] = _enemy_polygon(
            record["x"], record["y"], general=record["general"]
        )
        record["alive"] = True

    def _handle_player_hit() -> None:
        if destroyed_stack:
            revived = destroyed_stack.pop()
            _respawn_enemy(revived)
            if revived in kill_order:
                kill_order.remove(revived)
            action_var.set("You were hit! An invader regrouped. Keep fighting.")
        else:
            action_var.set("You were hit, but the fleet holds steady!")
        _update_status()

    def _destroy_enemy(enemy_id: str, explosion: bool = False) -> None:
        nonlocal pending_order, pending_duration, game_over
        record = enemy_data.get(enemy_id)
        if not record or not record["alive"]:
            return
        canvas.delete(record["item"])
        record["alive"] = False
        destroyed_stack.append(enemy_id)
        kill_order.append(enemy_id)
        if explosion:
            action_var.set(
                "Space charge unleashed a chain reaction! Nice shot."
            )
        _update_status()
        if len(kill_order) >= total_enemies:
            pending_order = kill_order[:]
            pending_duration = time.time() - start_time
            action_var.set(
                "Fleet neutralised! Click Use seed to apply the result."
            )
            accept_button.state(["!disabled"])
            game_over = True
            _cancel_timers()

    def _fire_player_shot(_event: "tk.Event") -> None:
        if len(player_shots) > 4 or game_over:
            return
        item = canvas.create_rectangle(
            player["x"] - 3,
            player["y"] - 20,
            player["x"] + 3,
            player["y"] - 30,
            fill="#ffffff",
            outline="",
        )
        player_shots.append({"item": item, "vy": -12})

    def _move_player(event: "tk.Event") -> None:
        if game_over:
            return
        player["x"] = max(40, min(canvas_width - 40, event.x))
        canvas.coords(
            player_item,
            player["x"] - 20,
            player["y"] + 20,
            player["x"],
            player["y"] - 20,
            player["x"] + 20,
            player["y"] + 20,
        )

    def _fire_enemy_shot(enemy_id: str) -> bool:
        record = enemy_data.get(enemy_id)
        if not record or not record["alive"]:
            return False
        item = canvas.create_rectangle(
            record["x"] - 4,
            record["y"] + 10,
            record["x"] + 4,
            record["y"] + 22,
            fill="#ff3366",
            outline="",
        )
        enemy_shots.append({"item": item, "vy": 6, "owner": enemy_id})
        return True

    def _spawn_enemy_shot_once() -> None:
        nonlocal last_shooter
        live_enemies = [
            eid for eid, rec in enemy_data.items() if rec["alive"]
        ]
        if not live_enemies:
            return
        per_cycle = 1 if len(live_enemies) < 3 else random.randint(1, 2)
        fired_ids: set[str] = set()
        for _ in range(per_cycle):
            candidates = [eid for eid in live_enemies if eid not in fired_ids]
            if not candidates:
                break
            if last_shooter in candidates and len(candidates) > 1:
                candidates = [
                    eid for eid in candidates if eid != last_shooter
                ]
            shooter_id = random.choice(candidates)
            if _fire_enemy_shot(shooter_id):
                fired_ids.add(shooter_id)
                last_shooter = shooter_id

    def _spawn_charge_once() -> None:
        if len(charges) >= 3:
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
        _spawn_enemy_shot_once()
        fire_handle = canvas.after(900, _enemy_fire_cycle)

    def _charge_cycle() -> None:
        nonlocal charge_handle
        if game_over:
            return
        _spawn_charge_once()
        charge_handle = canvas.after(7000, _charge_cycle)

    def _general_fire_cycle() -> None:
        nonlocal general_fire_handle
        if game_over:
            return
        if general_id:
            _fire_enemy_shot(general_id)
            _fire_enemy_shot(general_id)
        general_fire_handle = canvas.after(300, _general_fire_cycle)

    def _launch_bomb() -> None:
        item = canvas.create_polygon(
            player["x"] - 6,
            player["y"] - 10,
            player["x"],
            player["y"] - 30,
            player["x"] + 6,
            player["y"] - 10,
            fill="#ffda6b",
            outline="#fff1c7",
        )
        bombs.append({"item": item, "vy": -4, "chance": 0.45})

    def _handle_right_click(event: "tk.Event") -> None:
        nonlocal inventory_charge
        if game_over:
            return
        if inventory_charge:
            _launch_bomb()
            inventory_charge = False
            action_var.set("Space charge launched!")
            _update_status()
        else:
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

    def _update_entities() -> None:
        nonlocal general_direction, inventory_charge
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

        for shot in list(enemy_shots):
            canvas.move(shot["item"], 0, shot["vy"])
            coords = canvas.coords(shot["item"])
            if not coords or coords[3] > canvas_height:
                canvas.delete(shot["item"])
                enemy_shots.remove(shot)
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
                explosion = random.random() < bomb["chance"]
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
                                _destroy_enemy(eid)

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
                and not inventory_charge
                and charge_box[0] < player_box[2]
                and charge_box[2] > player_box[0]
                and charge_box[1] < player_box[3]
                and charge_box[3] > player_box[1]
            ):
                canvas.delete(charge["item"])
                charges.remove(charge)
                inventory_charge = True
                action_var.set(
                    "Space charge secured. Right-click to deploy it."
                )
                _update_status()

        if general_id:
            record = enemy_data.get(general_id)
            if record and record["alive"]:
                dx = 4 * general_direction
                record["x"] += dx
                canvas.move(record["item"], dx, 0)
                bbox = canvas.bbox(record["item"])
                if bbox:
                    if bbox[0] < 40 or bbox[2] > canvas_width - 40:
                        general_direction *= -1

    def _tick() -> None:
        nonlocal tick_handle
        if game_over:
            return
        _update_entities()
        tick_handle = canvas.after(40, _tick)

    def _start_timers() -> None:
        nonlocal tick_handle, fire_handle, charge_handle, general_fire_handle
        tick_handle = canvas.after(40, _tick)
        fire_handle = canvas.after(900, _enemy_fire_cycle)
        charge_handle = canvas.after(7000, _charge_cycle)
        general_fire_handle = canvas.after(300, _general_fire_cycle)

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
    window.protocol("WM_DELETE_WINDOW", _close_window)
    window.grab_set()
    _spawn_enemies()
    _update_status()
    _start_timers()
