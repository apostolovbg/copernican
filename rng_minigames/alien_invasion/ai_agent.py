"""Self-learning helper for the Alien Invasion mini-game."""

from __future__ import annotations

import math
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

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
DEFAULT_HIDDEN_UNITS = 12
MAX_HIDDEN_LAYERS = 10
MAX_LAYER_WIDTH = 64
OUTPUT_UNITS = 3
DEFAULT_HISTORY_LIMIT = 320
RUN_DURATION_DEFAULT = 300.0


def _normalize_hidden_layers(raw: Any) -> List[int]:
    values: List[int] = []

    def _extend_from_entry(entry: Any) -> None:
        if isinstance(entry, str) and "," in entry:
            for chunk in entry.split(","):
                _extend_from_entry(chunk.strip())
            return
        try:
            number = int(entry)
        except Exception:
            return
        number = max(1, min(MAX_LAYER_WIDTH, number))
        values.append(number)

    if isinstance(raw, str):
        for token in raw.split(","):
            token = token.strip()
            if token:
                _extend_from_entry(token)
    elif isinstance(raw, (list, tuple)):
        for entry in raw:
            _extend_from_entry(entry)
    elif raw is not None:
        _extend_from_entry(raw)

    if not values:
        values = [DEFAULT_HIDDEN_UNITS]
    if len(values) > MAX_HIDDEN_LAYERS:
        values = values[:MAX_HIDDEN_LAYERS]
    return values


def _initial_state(
    hidden_layers: Sequence[int] | None = None,
) -> Dict[str, Any]:
    layers = list(hidden_layers) if hidden_layers else [DEFAULT_HIDDEN_UNITS]
    return {
        "weights": {"aggression": 0.5, "caution": 0.5, "charge": 0.3},
        "network": _init_network(layers),
        "best_time": None,
        "runs": 0,
        "worlds_saved": 0,
        "worlds_lost": 0,
    }


def _init_network(hidden_layers: Sequence[int]) -> Dict[str, Any]:
    """Create a neural policy network with arbitrary hidden layers."""

    layers = list(hidden_layers) if hidden_layers else [DEFAULT_HIDDEN_UNITS]
    rng = random.Random()
    layer_sizes = [len(INPUT_FEATURES)] + layers + [OUTPUT_UNITS]
    weights: List[List[List[float]]] = []
    biases: List[List[float]] = []
    for idx in range(1, len(layer_sizes)):
        prev_size = layer_sizes[idx - 1]
        curr_size = layer_sizes[idx]
        weight_rows: List[List[float]] = []
        for _ in range(curr_size):
            spread = 0.25 if idx == len(layer_sizes) - 1 else 0.3
            weight_rows.append(
                [rng.uniform(-spread, spread) for _ in range(prev_size)]
            )
        weights.append(weight_rows)
        biases.append([rng.uniform(-0.1, 0.1) for _ in range(curr_size)])
    return {
        "input_size": len(INPUT_FEATURES),
        "hidden_layers": layers,
        "hidden_size": layers[-1],
        "output_size": OUTPUT_UNITS,
        "weights": weights,
        "biases": biases,
    }


class AlienInvasionAI:
    """Self-adjusting pilot that persists a lightweight neural policy."""

    def __init__(self, storage_dir: Path) -> None:
        storage_dir.mkdir(parents=True, exist_ok=True)
        self.settings = load_settings()
        self.hidden_layers = _normalize_hidden_layers(
            self.settings.get("hidden_units", DEFAULT_HIDDEN_UNITS)
        )
        self.hidden_units = self.hidden_layers[-1]
        self.history_limit = max(
            1, int(self.settings.get("history_limit", DEFAULT_HISTORY_LIMIT))
        )
        self.state_path = storage_dir / "alien_invasion_ai_state.yml"
        self.state: Dict[str, Any] = _initial_state(self.hidden_layers)
        self._history: List[Dict[str, Any]] = []
        self._time_pressure = 0.0
        self._intermediate_reward = 0.0
        self._kill_count = 0
        self._edge_streak = 0.0
        self.exploration_rate = float(
            self.settings.get("exploration_rate", 0.9)
        )
        tp_settings = self.settings.get("time_pressure", {})
        self.tp_base = float(tp_settings.get("base", 0.6))
        self.tp_scale = float(tp_settings.get("scale", 0.4))
        self.tp_exponent = float(tp_settings.get("exponent", 0.5))
        self.tp_fallback = float(tp_settings.get("fallback", 0.8))
        kill_cfg = self.settings.get("kill_reward", {})
        self.kill_base = float(kill_cfg.get("base", 0.7))
        self.kill_general_bonus = float(kill_cfg.get("general_bonus", 1.5))
        self.kill_increment = float(kill_cfg.get("increment", 0.15))
        self.kill_increment_cap = float(kill_cfg.get("max_increment", 4))
        respawn_cfg = self.settings.get("respawn_penalty", {})
        default_penalty = float(respawn_cfg.get("default", 0.3))
        self.respawn_penalty = {
            "lieutenant": float(
                respawn_cfg.get("lieutenant", default_penalty)
            ),
            "major": float(respawn_cfg.get("major", 0.5)),
            "colonel": float(respawn_cfg.get("colonel", 0.8)),
            "default": default_penalty,
        }
        self.edge_penalty_multiplier = float(
            self.settings.get("edge_penalty_multiplier", 8.0)
        )
        self.edge_streak_scale = float(
            self.settings.get("edge_streak_scale", 3.0)
        )
        self.edge_streak_decay = float(
            self.settings.get("edge_streak_decay", 1.5)
        )
        self.initial_weights = self.settings.get(
            "initial_weights",
            {"aggression": 0.7, "caution": 0.3, "charge": 0.5},
        )
        self.win_bonus = self.settings.get(
            "win_bonus", {"aggression": 0.2, "charge": 0.15, "caution": -0.05}
        )
        self.loss_caution_cap = float(
            self.settings.get("loss_caution_cap", 1.1)
        )
        self.kill_time_bonus = self.settings.get(
            "kill_time_bonus", {"multiplier": 2.5, "exponent": 1.5}
        )
        drought_cfg = self.settings.get(
            "kill_drought_penalty", {"multiplier": 1.7, "kills": 1}
        )
        self.kill_drought_multiplier = float(
            drought_cfg.get("multiplier", 1.7)
        )
        self.kill_drought_kills = float(drought_cfg.get("kills", 1))
        self.edge_streak_scale = float(
            self.settings.get("edge_streak_scale", 5.0)
        )
        self.max_run_duration = float(
            self.settings.get("run_duration_seconds", RUN_DURATION_DEFAULT)
        )
        self._load()
        self._ensure_network()
        if not self.state_path.exists():
            self._save()
        if self.state.get("runs", 0) == 0:
            self.state["weights"].update(self.initial_weights)

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

    def penalize_edge(self, amount: float) -> None:
        """Discourage the AI from hugging the screen edge."""

        if amount <= 0:
            return
        self._edge_streak += amount
        streak_penalty = self.edge_streak_scale * self._edge_streak
        penalty_value = (
            self.edge_penalty_multiplier * amount + streak_penalty
        )
        self._intermediate_reward -= penalty_value

    def cool_edge_streak(self, decay: float = 0.3) -> None:
        """Wind down the accumulated edge penalty when the AI leaves the wall."""

        decay_value = decay if decay is not None else self.edge_streak_decay
        self._edge_streak = max(0.0, self._edge_streak - decay_value)

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
        kill_rate = self._kill_count / max(duration if duration > 0 else 1.0, 1.0)
        multiplier = float(self.kill_time_bonus.get("multiplier", 1.0))
        exponent = float(self.kill_time_bonus.get("exponent", 1.0))
        reward += multiplier * (kill_rate**exponent)
        if self._kill_count <= self.kill_drought_kills:
            reward -= self.kill_drought_multiplier * max(duration, 1.0)
            if best_time is None or duration < best_time:
                self.state["best_time"] = duration
                reward += 4.0
            victories = self.state["worlds_saved"]
            weights["aggression"] += lr * (2.1 + victories * 0.08)
            weights["charge"] += lr * 0.55 * (1 + victories * 0.05)
            weights["caution"] -= lr * 0.4
            for key, bonus in self.win_bonus.items():
                if key in weights:
                    weights[key] += bonus
        else:
            self.state["worlds_lost"] = self.state.get("worlds_lost", 0) + 1
            defeats = self.state["worlds_lost"]
            penalty = 1 + defeats * 0.16
            time_penalty = min(6.0, duration / 35.0)
            reward = -7.5 * penalty - time_penalty - 2.0
            weights["caution"] += lr * 1.0 * penalty
            weights["caution"] = min(weights["caution"], self.loss_caution_cap)
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
            network = _init_network(self.hidden_layers)
        else:
            network = self._validate_or_reset_network(
                network, self.hidden_layers
            )
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

        self.state = _initial_state(self.hidden_layers)
        self._history.clear()
        try:
            if self.state_path.exists():
                self.state_path.unlink()
        except Exception:
            pass

    #
    # Neural helpers
    #

    def _ensure_network(self) -> None:
        if "network" not in self.state:
            self.state["network"] = _init_network(self.hidden_layers)
        else:
            self.state["network"] = self._validate_or_reset_network(
                self.state["network"], self.hidden_layers
            )

    def _validate_or_reset_network(
        self, network: Dict[str, Any], hidden_layers: Sequence[int]
    ) -> Dict[str, Any]:
        configured = (
            list(hidden_layers) if hidden_layers else [DEFAULT_HIDDEN_UNITS]
        )
        try:
            input_size = len(INPUT_FEATURES)
            output_size = OUTPUT_UNITS
            if {
                "weights",
                "biases",
            }.issubset(network.keys()):
                stored_layers = network.get("hidden_layers", configured)
                stored_layers = _normalize_hidden_layers(stored_layers)
                if stored_layers != configured:
                    raise ValueError("hidden layer mismatch")
                weights = network.get("weights")
                biases = network.get("biases")
                if not isinstance(weights, list) or not isinstance(
                    biases, list
                ):
                    raise ValueError("missing weight/bias matrices")
                if len(weights) != len(biases):
                    raise ValueError("weight/bias length mismatch")
                expected_sizes = [input_size] + configured + [output_size]
                if len(weights) != len(expected_sizes) - 1:
                    raise ValueError("layer count mismatch")
                sanitized_weights: List[List[List[float]]] = []
                sanitized_biases: List[List[float]] = []
                for idx, (out_size, in_size) in enumerate(
                    zip(expected_sizes[1:], expected_sizes[:-1])
                ):
                    layer_weights = weights[idx]
                    layer_biases = biases[idx]
                    if len(layer_weights) != out_size:
                        raise ValueError("invalid weight rows")
                    if len(layer_biases) != out_size:
                        raise ValueError("invalid bias rows")
                    sanitized_layer: List[List[float]] = []
                    for row in layer_weights:
                        if len(row) != in_size:
                            raise ValueError("invalid weight cols")
                        sanitized_layer.append([float(value) for value in row])
                    sanitized_biases.append(
                        [float(value) for value in layer_biases]
                    )
                    sanitized_weights.append(sanitized_layer)
                return {
                    "input_size": input_size,
                    "hidden_layers": configured,
                    "hidden_size": configured[-1],
                    "output_size": output_size,
                    "weights": sanitized_weights,
                    "biases": sanitized_biases,
                }
            # Legacy single-hidden-layer format
            if len(configured) != 1:
                raise ValueError("legacy network incompatible with layers")
            hidden_units = configured[0]
            legacy = {
                "w1",
                "b1",
                "w2",
                "b2",
            }
            if not legacy.issubset(network.keys()):
                raise ValueError("missing legacy weights")
            if (
                int(network.get("input_size", input_size)) != input_size
                or int(network.get("hidden_size", hidden_units))
                != hidden_units
                or int(network.get("output_size", output_size)) != output_size
            ):
                raise ValueError("legacy dimension mismatch")
            w1 = [
                [float(value) for value in row]
                for row in network.get("w1", [])
            ]
            if len(w1) != hidden_units:
                raise ValueError("invalid w1 rows")
            for row in w1:
                if len(row) != input_size:
                    raise ValueError("invalid w1 cols")
            w2 = [
                [float(value) for value in row]
                for row in network.get("w2", [])
            ]
            if len(w2) != output_size:
                raise ValueError("invalid w2 rows")
            for row in w2:
                if len(row) != hidden_units:
                    raise ValueError("invalid w2 cols")
            b1 = [float(value) for value in network.get("b1", [])]
            b2 = [float(value) for value in network.get("b2", [])]
            if len(b1) != hidden_units or len(b2) != output_size:
                raise ValueError("invalid legacy bias length")
            return {
                "input_size": input_size,
                "hidden_layers": configured,
                "hidden_size": hidden_units,
                "output_size": output_size,
                "weights": [w1, w2],
                "biases": [b1, b2],
            }
        except Exception:
            return _init_network(configured)

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
        if len(self._history) > self.history_limit:
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
        weights = network["weights"]
        biases = network["biases"]
        prev_activation = [float(value) for value in features]
        layer_states: List[Dict[str, Any]] = []
        for idx, weight_matrix in enumerate(weights):
            layer_biases = biases[idx]
            raw_outputs: List[float] = []
            for neuron_idx, weight_row in enumerate(weight_matrix):
                total = layer_biases[neuron_idx]
                for prev_idx, value in enumerate(prev_activation):
                    total += weight_row[prev_idx] * value
                raw_outputs.append(total)
            if idx == len(weights) - 1:
                activated = raw_outputs[:]
            else:
                activated = [math.tanh(value) for value in raw_outputs]
            layer_states.append(
                {
                    "raw": raw_outputs,
                    "activated": activated,
                    "input": prev_activation,
                    "is_output": idx == len(weights) - 1,
                }
            )
            prev_activation = activated
        final_raw = (
            layer_states[-1]["raw"] if layer_states else [0.0] * OUTPUT_UNITS
        )
        move = math.tanh(final_raw[0])
        shoot = 1.0 / (1.0 + math.exp(-final_raw[1]))
        charge = 1.0 / (1.0 + math.exp(-final_raw[2]))
        return {
            "layers": layer_states,
            "raw_outputs": final_raw,
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
        recent_history = self._history[-self.history_limit :]
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
        weights = network["weights"]
        biases = network["biases"]
        layers_info = forward["layers"]
        if not weights or not layers_info:
            return
        layer_deltas: List[List[float]] = [
            [0.0 for _ in layer["activated"]] for layer in layers_info
        ]
        layer_deltas[-1] = output_deltas
        for layer_idx in range(len(weights) - 2, -1, -1):
            next_weights = weights[layer_idx + 1]
            next_delta = layer_deltas[layer_idx + 1]
            activations = layers_info[layer_idx]["activated"]
            current_delta: List[float] = []
            for neuron_idx, neuron_activation in enumerate(activations):
                influence = 0.0
                for next_idx, next_weights_row in enumerate(next_weights):
                    influence += (
                        next_weights_row[neuron_idx] * next_delta[next_idx]
                    )
                current_delta.append(influence * (1 - neuron_activation**2))
            layer_deltas[layer_idx] = current_delta

        for layer_idx, delta_vec in enumerate(layer_deltas):
            prev_activation = (
                sample["features"]
                if layer_idx == 0
                else layers_info[layer_idx - 1]["activated"]
            )
            weight_matrix = weights[layer_idx]
            bias_vec = biases[layer_idx]
            for neuron_idx, delta_value in enumerate(delta_vec):
                for prev_idx, value in enumerate(prev_activation):
                    update = lr * reward * delta_value * value
                    weight_matrix[neuron_idx][prev_idx] = self._clamp(
                        weight_matrix[neuron_idx][prev_idx] + update
                    )
                bias_vec[neuron_idx] = self._clamp(
                    bias_vec[neuron_idx] + lr * reward * delta_value
                )

    @staticmethod
    def _clamp(value: float, limit: float = 5.0) -> float:
        return float(max(-limit, min(limit, value)))
