"""CombatEnv(gym.Env) — reset/step/observation for the 1v1 combat simulation."""
import math
from pathlib import Path

import gymnasium as gym
import numpy as np
import yaml
from gymnasium import spaces

from . import ballistics, geometry, spawner

_CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"

# move_direction: 0 = stay, 1..8 = 8 compass directions (unit vectors, normalized in _move_soldier)
MOVE_VECTORS = [
    (0.0, 0.0),    # stay
    (0.0, 1.0),    # N
    (1.0, 1.0),    # NE
    (1.0, 0.0),    # E
    (1.0, -1.0),   # SE
    (0.0, -1.0),   # S
    (-1.0, -1.0),  # SW
    (-1.0, 0.0),   # W
    (-1.0, 1.0),   # NW
]
N_MOVE_DIRECTIONS = len(MOVE_VECTORS)
N_FIRE_STATES = 2


def _load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


class CombatEnv(gym.Env):
    metadata = {"render_modes": [None, "human"]}

    def __init__(self, env_config=None, reward_config=None, render_mode=None,
                 enemy_fire_enabled=True):
        super().__init__()

        if env_config is None:
            env_config = _load_yaml(_CONFIG_DIR / "env_default.yaml")
        if reward_config is None:
            reward_config = _load_yaml(_CONFIG_DIR / "reward_default.yaml")

        self.render_mode = render_mode
        self.enemy_fire_enabled = bool(enemy_fire_enabled)

        self.map_width = float(env_config["map"]["width"])
        self.map_height = float(env_config["map"]["height"])
        self.bin_size_degrees = int(env_config["bin_size_degrees"])
        self.n_bins = 360 // self.bin_size_degrees
        self.timestep_duration_s = float(env_config["timestep_duration_s"])
        self.max_timesteps = int(env_config["episode"]["max_timesteps"])
        self.min_spawn_separation = float(env_config["episode"]["min_spawn_separation"])
        self.move_step_size = float(env_config["movement"]["step_size"])

        gun = env_config["gun"]
        self.max_ammo = int(gun["max_ammo"])
        self.max_range = float(gun["max_range"])
        self.fire_cooldown_steps = int(gun["fire_cooldown_steps"])
        self.accuracy_table = gun["accuracy_table"]
        self.damage_table = gun["damage_table"]

        self.soldier_max_hp = float(env_config["soldier"]["max_hp"])
        self.enemy_max_hp = float(env_config["enemy"]["max_hp"])

        self.step_penalty = float(reward_config["step_penalty"])
        self.hit_reward_scale = float(reward_config["hit_reward_scale"])
        self.damage_taken_penalty_scale = float(reward_config["damage_taken_penalty_scale"])
        self.empty_gun_fire_penalty = float(reward_config["empty_gun_fire_penalty"])
        self.orientation_shaping_bonus_weight = float(reward_config["orientation_shaping_bonus_weight"])
        self.terminal_win = float(reward_config["terminal"]["win"])
        self.terminal_lose = float(reward_config["terminal"]["lose"])
        self.terminal_timeout = float(reward_config["terminal"]["timeout"])

        self.action_space = spaces.Discrete(N_MOVE_DIRECTIONS * N_FIRE_STATES * self.n_bins)

        obs_dim = 2 + 4 + 4 + 1 + 1 + self.n_bins + self.n_bins
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self.soldier = None
        self.enemy = None
        self.t = 0
        self.last_hit_events = []

    # -- Gymnasium API -----------------------------------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        soldier_pos, enemy_pos = spawner.spawn_positions(
            self.map_width, self.map_height, self.min_spawn_separation, self.np_random
        )
        soldier_orientation_bin = spawner.spawn_orientation_bin(self.n_bins, self.np_random)

        self.soldier = {
            "position": np.array(soldier_pos, dtype=np.float64),
            "orientation_bin": soldier_orientation_bin,
            "hp": float(self.soldier_max_hp),
            "ammo": int(self.max_ammo),
            "cooldown_remaining": 0,
        }
        self.enemy = {
            "position": np.array(enemy_pos, dtype=np.float64),
            "orientation_bin": 0,
            "hp": float(self.enemy_max_hp),
            "ammo": int(self.max_ammo),
            "cooldown_remaining": 0,
        }
        self._update_enemy_orientation()
        self.t = 0
        self.last_hit_events = []  # [(shooter_pos, target_pos), ...] for the renderer's on-hit tracer

        obs = self._get_obs()
        info = {}
        return obs, info

    def step(self, action):
        move_dir, fire, orientation_bin = self._decode_action(action)

        self.soldier["cooldown_remaining"] = max(0, self.soldier["cooldown_remaining"] - 1)
        self.enemy["cooldown_remaining"] = max(0, self.enemy["cooldown_remaining"] - 1)

        self.soldier["orientation_bin"] = orientation_bin
        self._move_soldier(move_dir)
        self._update_enemy_orientation()

        soldier_hit, soldier_damage, soldier_ammo_consumed, soldier_fire_executed = (
            self._resolve_soldier_fire(fire)
        )
        enemy_hit, enemy_damage, _, _ = self._resolve_enemy_fire()

        self.last_hit_events = []
        if soldier_hit:
            self.last_hit_events.append((tuple(self.soldier["position"]), tuple(self.enemy["position"])))
        if enemy_hit:
            self.last_hit_events.append((tuple(self.enemy["position"]), tuple(self.soldier["position"])))

        self.enemy["hp"] = max(0.0, self.enemy["hp"] - soldier_damage)
        self.soldier["hp"] = max(0.0, self.soldier["hp"] - enemy_damage)

        self.t += 1

        terminated, outcome = self._check_terminated()

        empty_gun_fire = soldier_fire_executed and not soldier_ammo_consumed
        reward = self._get_reward(
            soldier_damage_dealt=soldier_damage,
            enemy_damage_taken=enemy_damage,
            empty_gun_fire=empty_gun_fire,
            soldier_facing_enemy=self._soldier_facing_enemy(),
            outcome=outcome,
        )

        obs = self._get_obs()
        info = {"outcome": outcome, "soldier_hit": soldier_hit, "enemy_hit": enemy_hit}
        truncated = False
        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode != "human":
            return None
        from rendering import pygame_renderer
        return pygame_renderer.render(self)

    def close(self):
        if self.render_mode == "human":
            from rendering import pygame_renderer
            pygame_renderer.close()

    # -- Internals -----------------------------------------------------------

    def _decode_action(self, action):
        n_bins = self.n_bins
        orientation_bin = action % n_bins
        rem = action // n_bins
        fire = rem % 2
        move_dir = rem // 2
        return move_dir, fire, orientation_bin

    def _move_soldier(self, move_dir):
        dx, dy = MOVE_VECTORS[move_dir]
        norm = math.hypot(dx, dy)
        if norm > 0:
            dx, dy = dx / norm, dy / norm
        new_x = float(np.clip(self.soldier["position"][0] + dx * self.move_step_size, 0.0, self.map_width))
        new_y = float(np.clip(self.soldier["position"][1] + dy * self.move_step_size, 0.0, self.map_height))
        self.soldier["position"] = np.array([new_x, new_y], dtype=np.float64)

    def _update_enemy_orientation(self):
        bearing = geometry.bearing(self.enemy["position"], self.soldier["position"])
        self.enemy["orientation_bin"] = geometry.angle_to_bin(bearing, self.bin_size_degrees)

    def _soldier_facing_enemy(self):
        bearing = geometry.bearing(self.soldier["position"], self.enemy["position"])
        true_bin = geometry.angle_to_bin(bearing, self.bin_size_degrees)
        return ballistics.check_los(self.soldier["orientation_bin"], true_bin)

    def _table_config(self):
        return {
            "bin_size_degrees": self.bin_size_degrees,
            "accuracy_table": self.accuracy_table,
            "damage_table": self.damage_table,
        }

    def _resolve_soldier_fire(self, fire):
        if not fire or self.soldier["cooldown_remaining"] > 0:
            return False, 0.0, False, False

        hit, damage, ammo_consumed = ballistics.resolve_shot(
            self.soldier, self.enemy, self._table_config(), rng=self.np_random
        )
        if ammo_consumed:
            self.soldier["ammo"] -= 1
            self.soldier["cooldown_remaining"] = self.fire_cooldown_steps
        return hit, damage, ammo_consumed, True

    def _resolve_enemy_fire(self):
        if not self.enemy_fire_enabled or self.enemy["cooldown_remaining"] > 0:
            return False, 0.0, False, False
        if geometry.distance(self.enemy["position"], self.soldier["position"]) > self.max_range:
            return False, 0.0, False, False

        hit, damage, ammo_consumed = ballistics.resolve_shot(
            self.enemy, self.soldier, self._table_config(), rng=self.np_random
        )
        if ammo_consumed:
            self.enemy["ammo"] -= 1
            self.enemy["cooldown_remaining"] = self.fire_cooldown_steps
        return hit, damage, ammo_consumed, True

    def _check_terminated(self):
        # Enemy death takes priority: a simultaneous double-kill counts as a win.
        if self.enemy["hp"] <= 0.0:
            return True, "win"
        if self.soldier["hp"] <= 0.0:
            return True, "loss"
        if self.t >= self.max_timesteps:
            return True, "timeout"
        return False, None

    def _get_reward(self, soldier_damage_dealt, enemy_damage_taken, empty_gun_fire,
                     soldier_facing_enemy, outcome):
        reward = self.step_penalty
        reward += self.hit_reward_scale * soldier_damage_dealt
        reward -= self.damage_taken_penalty_scale * enemy_damage_taken
        if empty_gun_fire:
            reward += self.empty_gun_fire_penalty
        if soldier_facing_enemy:
            reward += self.orientation_shaping_bonus_weight
        if outcome == "win":
            reward += self.terminal_win
        elif outcome == "loss":
            reward += self.terminal_lose
        elif outcome == "timeout":
            reward += self.terminal_timeout
        return float(reward)

    def _get_obs(self):
        dx = self.enemy["position"][0] - self.soldier["position"][0]
        dy = self.enemy["position"][1] - self.soldier["position"][1]
        dist = geometry.distance(self.soldier["position"], self.enemy["position"])
        accuracy_at_dist = ballistics.accuracy_lookup(dist, self.accuracy_table)

        soldier_orientation_onehot = np.zeros(self.n_bins, dtype=np.float32)
        soldier_orientation_onehot[self.soldier["orientation_bin"]] = 1.0
        enemy_orientation_onehot = np.zeros(self.n_bins, dtype=np.float32)
        enemy_orientation_onehot[self.enemy["orientation_bin"]] = 1.0

        obs = np.concatenate([
            np.array([dx, dy], dtype=np.float32),
            np.array([
                self.soldier["ammo"], self.max_range,
                self.soldier["cooldown_remaining"], accuracy_at_dist,
            ], dtype=np.float32),
            np.array([
                self.enemy["ammo"], self.max_range,
                self.enemy["cooldown_remaining"], accuracy_at_dist,
            ], dtype=np.float32),
            np.array([self.soldier["hp"], self.enemy["hp"]], dtype=np.float32),
            soldier_orientation_onehot,
            enemy_orientation_onehot,
        ])
        return obs.astype(np.float32)
