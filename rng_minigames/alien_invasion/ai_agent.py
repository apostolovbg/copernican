"""Self-learning helper for the Alien Invasion mini-game."""

from __future__ import annotations

import math
import random
import time
from pathlib import Path
from typing import Any, Dict, List

import yaml

from .ai_config import load_settings

INPUT_FEATURES = (
    "target_offset",
    "alignment_norm",
    "incoming_bias",
    "incoming_direction",
    "cluster_score",
    "general_offset",
    "charge_ratio",
    "player_pressure",
    "incoming_density",
    "edge_bias",
    "general_distance",
    "time_remaining_fraction",
)
HIDDEN_UNITS = 12
OUTPUT_UNITS = 3
HISTORY_LIMIT = 320
RUN_DURATION_DEFAULT = 300.0


def _initial_state() -> Dict[str, Any]:
    return {
        "weights": {"aggression": 0.5, "caution": 0.5, "charge": 0.3},
        "network": _init_network(),
        "best_time": None,
        "runs": 0,
        "worlds_saved": 0,
        "worlds_lost": 0,
    }


def _init_network() -> Dict[str, Any]:
    """Create a small neural policy network."""

    rng = random.Random()
    network = {
        "input_size": len(INPUT_FEATURES),
        "hidden_size": HIDDEN_UNITS,
        "output_size": OUTPUT_UNITS,
        "w1": [
            [rng.uniform(-0.3, 0.3) for _ in range(len(INPUT_FEATURES))]
            for _ in range(HIDDEN_UNITS)
        ],
        "b1": [rng.uniform(-0.1, 0.1) for _ in range(HIDDEN_UNITS)],
        "w2": [
            [rng.uniform(-0.25, 0.25) for _ in range(HIDDEN_UNITS)]
            for _ in range(OUTPUT_UNITS)
        ],
        "b2": [0.0 for _ in range(OUTPUT_UNITS)],
    }
    return network


class AlienInvasionAI:
    """Self-adjusting pilot that persists a lightweight neural policy."""

    def __init__(self, storage_dir: Path) -> None:
        self.state_path = storage_dir / "alien_invasion_ai_state.yml"
        self.state: Dict[str, Any] = _initial_state()
        self._history: List[Dict[str, Any]] = []
        self._time_pressure = 0.0
        self._intermediate_reward = 0.0
        self._kill_count = 0
        self.settings = load_settings()
        self.exploration_rate = float(
            self.settings.get("exploration_rate", 0.75)
        )
        tp_settings = self.settings.get("time_pressure", {})
        self.tp_base = float(tp_settings.get("base", 0.6))
        self.tp_scale = float(tp_settings.get("scale", 0.4))
        self.tp_exponent = float(tp_settings.get("exponent", 0.5))
        self.tp_fallback = float(tp_settings.get("fallback", 0.8))
        kill_cfg = self.settings.get("kill_reward", {})
        self.kill_base = float(kill_cfg.get("base", 0.7))
        self.kill_general_bonus = float(kill_cfg.get("general_bonus", 1.5))
        self.kill_increment = float(kill_cfg.get("increment", 0.08))
        self.kill_increment_cap = float(kill_cfg.get("max_increment", 2.5))
        respawn_cfg = self.settings.get("respawn_penalty", {})
        default_penalty = float(respawn_cfg.get("default", 0.25))
        self.respawn_penalty = {
            "lieutenant": float(
                respawn_cfg.get("lieutenant", default_penalty)
            ),
            "major": float(respawn_cfg.get("major", 0.35)),
            "colonel": float(respawn_cfg.get("colonel", 0.45)),
            "default": default_penalty,
        }
        self.max_run_duration = float(
            self.settings.get("run_duration_seconds", RUN_DURATION_DEFAULT)
        )
        self._load()

    def decide(self, snapshot: Dict[str, Any]) -> Dict[str, Any]:
        """Return movement/shoot/charge decisions based on the snapshot."""

        self._ensure_network()
        features = self._extract_features(snapshot)
        time_fraction = snapshot.get("time_remaining_fraction")
        if isinstance(time_fraction, (int, float)):
            self._time_pressure = self._compute_time_pressure_value(
                float(time_fraction)
            )
        else:
            self._time_pressure = self.tp_fallback
        feature_vector = self._feature_vector(features)
        forward = self._forward(feature_vector)
        weights = self.state["weights"]

        edge_correction = -features["edge_bias"] * 0.35
        general_proximity = max(0.0, 0.8 - features["general_distance"] * 3.0)
        general_correction = (
            -features["general_offset"]
            * general_proximity
            * (0.5 + weights["caution"] * 0.3)
        )
        urgency = self._time_pressure
        move_value = math.tanh(
            forward["move"]
            + features["target_offset"] * weights["aggression"] * 0.4
            - features["incoming_bias"] * weights["caution"] * 0.3
            + edge_correction
            + general_correction
            + urgency * 1.0
        )
        move_dir = max(-1.0, min(1.0, move_value))
        if abs(move_dir) < 0.05:
            move_dir = 0.0

        shoot_bias = (
            weights["aggression"] * 0.15
            - weights["caution"] * 0.08
            + urgency * 0.7
        )
        shoot_prob = min(0.99, max(0.01, forward["shoot"] + shoot_bias))
        shoot = shoot_prob > random.random()

        charge_bias = weights["charge"] * 0.2 + urgency * 0.45
        charge_prob = min(0.98, max(0.01, forward["charge"] + charge_bias))
        charge = (
            snapshot.get("charges", 0) > 0 and charge_prob > random.random()
        )

        if random.random() < self.exploration_rate:  # exploration
            move_dir = random.choice([-1.0, 0.0, 1.0])
        if random.random() < self.exploration_rate:
            shoot = not shoot
        if random.random() < self.exploration_rate * 0.75:
            charge = not charge

        self._remember_sample(
            feature_vector,
            move_dir,
            1 if shoot else 0,
            1 if charge else 0,
        )
        return {"move": move_dir, "shoot": shoot, "charge": charge}

    def begin_run(self) -> None:
        """Reset incremental reward trackers at the start of each run."""

        self._intermediate_reward = 0.0
        self._kill_count = 0

    def reward_enemy_destroyed(
        self, rank: str, *, general: bool = False
    ) -> None:
        """Provide a small carrot whenever a hostile is eliminated."""

        self._kill_count += 1
        base = self.kill_base + (self.kill_general_bonus if general else 0.0)
        increment = min(
            self._kill_count * self.kill_increment, self.kill_increment_cap
        )
        self._intermediate_reward += base + increment

    def penalize_enemy_respawned(self, rank: str) -> None:
        """Apply a stick when a destroyed enemy respawns."""

        penalty = self.respawn_penalty.get(
            rank, self.respawn_penalty.get("default", 0.25)
        )
        self._intermediate_reward -= penalty

    def _compute_time_pressure_value(self, fraction: float) -> float:
        clamped = max(0.0, min(1.0, fraction))
        return (
            self.tp_base + (1.0 - clamped) ** self.tp_exponent * self.tp_scale
        )

    def record_run(self, *, success: bool, duration: float) -> None:
        """Adjust weights and train the neural policy based on run outcome."""

        self._ensure_network()
        self.state["runs"] += 1
        weights = self.state["weights"]
        lr = 0.5
        best_time = self.state.get("best_time")
        duration = max(0.0, duration)
        if success:
            self.state["worlds_saved"] = self.state.get("worlds_saved", 0) + 1
            normalized = 1.0 - min(duration, self.max_run_duration) / max(
                self.max_run_duration, 1.0
            )
            speed_bonus = 6.0 * (0.3 + 0.7 * normalized)
            streak_bonus = 1.0 + self.state["worlds_saved"] * 0.06
            reward = 12.0 * streak_bonus + speed_bonus + 3.5
            if best_time is None or duration < best_time:
                self.state["best_time"] = duration
                reward += 4.0
            victories = self.state["worlds_saved"]
            weights["aggression"] += lr * (2.1 + victories * 0.08)
            weights["charge"] += lr * 0.55 * (1 + victories * 0.05)
            weights["caution"] -= lr * 0.4
        else:
            self.state["worlds_lost"] = self.state.get("worlds_lost", 0) + 1
            defeats = self.state["worlds_lost"]
            penalty = 1 + defeats * 0.16
            time_penalty = min(6.0, duration / 35.0)
            reward = -7.5 * penalty - time_penalty - 2.0
            weights["caution"] += lr * 1.0 * penalty
            weights["aggression"] -= lr * 0.85 * penalty
            weights["charge"] -= lr * 0.25
        reward += self._intermediate_reward
        self._intermediate_reward = 0.0
        for key in ("aggression", "caution", "charge"):
            weights[key] = float(min(1.5, max(0.05, weights[key])))
        self._train_history(reward)
        self._history.clear()
        self._save()

    #
    # Internal helpers
    #

    def _extract_features(self, snapshot: Dict[str, Any]) -> Dict[str, float]:
        player_x = snapshot.get("player_x", 0.0)
        canvas_width = snapshot.get("canvas_width", 1) or 1
        target_offset = 0.0
        alignment = 999.0
        enemies: List[Dict[str, Any]] = snapshot.get("enemies", [])
        if enemies:
            target = min(
                enemies,
                key=lambda entry: abs(entry["x"] - player_x)
                + entry["y"] * 0.3,
            )
            alignment = abs(target["x"] - player_x)
            target_offset = (target["x"] - player_x) / canvas_width
        incoming_bias = 0.0
        incoming_dir = 0.0
        for shot in snapshot.get("incoming", []):
            horizontal_gap = shot["x"] - player_x
            vertical_gap = max(shot["y"], 1)
            if abs(horizontal_gap) < 90:
                incoming_bias += 1 / vertical_gap
                incoming_dir += -1 if horizontal_gap > 0 else 1
        cluster_score = min(1.0, len(enemies) / 20)
        general_x = snapshot.get("general_x", player_x)
        charge_ratio = min(1.0, snapshot.get("charges", 0) / 3.0)
        player_pressure = min(1.0, len(snapshot.get("player_shots", [])) / 5.0)
        incoming_density = min(1.0, len(snapshot.get("incoming", [])) / 6.0)
        edge_bias = (player_x - canvas_width / 2) / max(canvas_width / 2, 1)
        general_distance = abs(general_x - player_x) / max(canvas_width, 1)
        time_remaining_fraction = snapshot.get("time_remaining_fraction")
        if time_remaining_fraction is None:
            time_remaining_fraction = 1.0
        return {
            "target_offset": target_offset + incoming_dir * 0.05,
            "alignment_norm": alignment / max(canvas_width / 2, 1),
            "incoming_bias": math.tanh(incoming_bias),
            "incoming_direction": max(-1.0, min(1.0, incoming_dir * 0.2)),
            "cluster_score": cluster_score,
            "general_offset": (general_x - player_x) / max(canvas_width, 1),
            "charge_ratio": charge_ratio,
            "player_pressure": player_pressure,
            "incoming_density": incoming_density,
            "edge_bias": edge_bias,
            "general_distance": general_distance,
            "time_remaining_fraction": time_remaining_fraction,
        }

    def _load(self) -> None:
        if not self.state_path.exists():
            return
        try:
            data = yaml.safe_load(self.state_path.read_text()) or {}
        except Exception:
            return
        if "weights" in data:
            self.state["weights"].update(data["weights"])
        network = data.get("network")
        if not isinstance(network, dict):
            network = _init_network()
        else:
            network = self._validate_or_reset_network(network)
        self.state["network"] = network
        if "best_time" in data:
            self.state["best_time"] = data["best_time"]
        if "runs" in data:
            self.state["runs"] = data["runs"]
        if "worlds_saved" in data:
            self.state["worlds_saved"] = data["worlds_saved"]
        if "worlds_lost" in data:
            self.state["worlds_lost"] = data["worlds_lost"]

    def _save(self) -> None:
        payload = {
            "weights": self.state["weights"],
            "network": self.state["network"],
            "best_time": self.state["best_time"],
            "runs": self.state["runs"],
            "worlds_saved": self.state.get("worlds_saved", 0),
            "worlds_lost": self.state.get("worlds_lost", 0),
            "updated": time.time(),
        }
        try:
            self.state_path.write_text(yaml.safe_dump(payload))
        except Exception:
            pass

    def forget(self) -> None:
        """Wipe the saved weights so the AI restarts from scratch."""

        self.state = _initial_state()
        self._history.clear()
        try:
            if self.state_path.exists():
                self.state_path.unlink()
        except Exception:
            pass
        self._save()

    #
    # Neural helpers
    #

    def _ensure_network(self) -> None:
        if "network" not in self.state:
            self.state["network"] = _init_network()
        else:
            self.state["network"] = self._validate_or_reset_network(
                self.state["network"]
            )

    def _validate_or_reset_network(
        self, network: Dict[str, Any]
    ) -> Dict[str, Any]:
        try:
            if (
                int(network.get("input_size")) != len(INPUT_FEATURES)
                or int(network.get("hidden_size")) != HIDDEN_UNITS
                or int(network.get("output_size")) != OUTPUT_UNITS
            ):
                raise ValueError("network dimensions mismatch")
            w1 = [
                [float(value) for value in row]
                for row in network.get("w1", [])
            ]
            if len(w1) != HIDDEN_UNITS:
                raise ValueError("invalid w1 rows")
            for row in w1:
                if len(row) != len(INPUT_FEATURES):
                    raise ValueError("invalid w1 cols")
            w2 = [
                [float(value) for value in row]
                for row in network.get("w2", [])
            ]
            if len(w2) != OUTPUT_UNITS:
                raise ValueError("invalid w2 rows")
            for row in w2:
                if len(row) != HIDDEN_UNITS:
                    raise ValueError("invalid w2 cols")
            b1 = [float(value) for value in network.get("b1", [])]
            b2 = [float(value) for value in network.get("b2", [])]
            if len(b1) != HIDDEN_UNITS or len(b2) != OUTPUT_UNITS:
                raise ValueError("invalid bias length")
            return {
                "input_size": len(INPUT_FEATURES),
                "hidden_size": HIDDEN_UNITS,
                "output_size": OUTPUT_UNITS,
                "w1": w1,
                "b1": b1,
                "w2": w2,
                "b2": b2,
            }
        except Exception:
            return _init_network()

    def _feature_vector(self, features: Dict[str, float]) -> List[float]:
        return [float(features[key]) for key in INPUT_FEATURES]

    def _remember_sample(
        self,
        features: List[float],
        move: float,
        shoot: int,
        charge: int,
    ) -> None:
        if not features:
            return
        self._history.append(
            {
                "features": features,
                "move": float(move),
                "shoot": int(shoot),
                "charge": int(charge),
            }
        )
        if len(self._history) > HISTORY_LIMIT:
            self._history.pop(0)

    def _forward(self, features: List[float]) -> Dict[str, float]:
        result = self._forward_internal(features)
        return {
            "move": result["move"],
            "shoot": result["shoot"],
            "charge": result["charge"],
        }

    def _forward_internal(self, features: List[float]) -> Dict[str, Any]:
        network = self.state["network"]
        hidden: List[float] = []
        for i in range(HIDDEN_UNITS):
            total = network["b1"][i]
            for j, value in enumerate(features):
                total += network["w1"][i][j] * value
            hidden.append(math.tanh(total))
        raw_outputs: List[float] = []
        for k in range(OUTPUT_UNITS):
            total = network["b2"][k]
            for i, h_val in enumerate(hidden):
                total += network["w2"][k][i] * h_val
            raw_outputs.append(total)
        move = math.tanh(raw_outputs[0])
        shoot = 1.0 / (1.0 + math.exp(-raw_outputs[1]))
        charge = 1.0 / (1.0 + math.exp(-raw_outputs[2]))
        return {
            "hidden": hidden,
            "outputs": raw_outputs,
            "move": move,
            "shoot": shoot,
            "charge": charge,
        }

    def _train_history(self, reward: float) -> None:
        if not self._history:
            return
        magnitude = max(0.2, min(6.0, abs(reward)))
        sign = 1 if reward >= 0 else -1
        adjusted = sign * magnitude
        base_lr = 0.03 + min(0.07, magnitude * 0.01)
        passes = 1
        if magnitude > 2.5:
            passes += 1
        if magnitude > 4.5:
            passes += 1
        recent_history = self._history[-HISTORY_LIMIT:]
        history_len = len(recent_history)
        for _ in range(passes):
            for index, sample in enumerate(recent_history):
                decay = 0.982 ** (history_len - index)
                self._train_sample(sample, adjusted * decay, base_lr)

    def _train_sample(
        self, sample: Dict[str, Any], reward: float, lr: float
    ) -> None:
        forward = self._forward_internal(sample["features"])
        move_target = sample["move"] if reward >= 0 else -sample["move"]
        move_delta = (move_target - forward["move"]) * (
            1 - forward["move"] ** 2
        )

        shoot_target = sample["shoot"] if reward >= 0 else 1 - sample["shoot"]
        shoot_error = shoot_target - forward["shoot"]
        shoot_delta = shoot_error * forward["shoot"] * (1 - forward["shoot"])

        charge_target = (
            sample["charge"] if reward >= 0 else 1 - sample["charge"]
        )
        charge_error = charge_target - forward["charge"]
        charge_delta = (
            charge_error * forward["charge"] * (1 - forward["charge"])
        )

        output_deltas = [move_delta, shoot_delta, charge_delta]
        network = self.state["network"]

        for out_idx in range(OUTPUT_UNITS):
            for h_idx in range(HIDDEN_UNITS):
                delta = (
                    lr
                    * reward
                    * output_deltas[out_idx]
                    * forward["hidden"][h_idx]
                )
                network["w2"][out_idx][h_idx] = self._clamp(
                    network["w2"][out_idx][h_idx] + delta
                )
            network["b2"][out_idx] = self._clamp(
                network["b2"][out_idx] + lr * reward * output_deltas[out_idx]
            )

        hidden_deltas: List[float] = []
        for h_idx in range(HIDDEN_UNITS):
            influence = sum(
                network["w2"][out_idx][h_idx] * output_deltas[out_idx]
                for out_idx in range(OUTPUT_UNITS)
            )
            hidden_deltas.append(
                influence * (1 - forward["hidden"][h_idx] ** 2)
            )

        for h_idx in range(HIDDEN_UNITS):
            for in_idx, value in enumerate(sample["features"]):
                delta = lr * reward * hidden_deltas[h_idx] * value
                network["w1"][h_idx][in_idx] = self._clamp(
                    network["w1"][h_idx][in_idx] + delta
                )
            network["b1"][h_idx] = self._clamp(
                network["b1"][h_idx] + lr * reward * hidden_deltas[h_idx]
            )

    @staticmethod
    def _clamp(value: float, limit: float = 5.0) -> float:
        return float(max(-limit, min(limit, value)))
