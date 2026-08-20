"""CombatEnv(gym.Env): a 1v1 top-down combat simulation — one learning "soldier"
against a scripted "enemy" on a rectangular map.

Each tick the soldier picks one of N_MOVE_DIRECTIONS * N_FIRE_STATES * n_bins
discrete actions (move direction, fire or not, and the orientation bin to face),
moves, optionally fires, and the scripted enemy turns to face the soldier's true
bearing and fires back if enemy_fire_enabled. Line-of-sight is exact-bin
matching (ballistics.check_los): a shot only has a chance to hit if the
shooter's orientation bin equals the target's true-bearing bin. Hit chance and
damage are both range-interpolated lookup tables (env_config["gun"]).

The enemy's toughness is tunable independently of its base stats via
set_enemy_difficulty() (reaction_interval_steps / fire_cooldown_steps /
fire_enabled) — this is what training/train.py's curriculum ramps up over the
course of a run, from "can't shoot back" to full difficulty.

Config is split across two YAML sections, loaded from config/ by default:
env_config (config/sim_default.yaml's env:) controls map/gun/HP/movement, and
reward_config (config/training_default.yaml's reward:) controls reward shaping
(see _get_reward). Either can be overridden by passing an already-loaded dict.

Episodes end in "win" (enemy HP -> 0, including a simultaneous double-kill),
"loss" (soldier HP -> 0), or "timeout" (max_timesteps reached with both alive).
"""
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


def _load_sim_defaults():
    return _load_yaml(_CONFIG_DIR / "sim_default.yaml")


def _load_training_defaults():
    return _load_yaml(_CONFIG_DIR / "training_default.yaml")


class CombatEnv(gym.Env):
    metadata = {"render_modes": [None, "human"]}

    def __init__(self, env_config=None, reward_config=None, render_config=None, render_mode=None,
                 enemy_fire_enabled=True, enemy_reaction_interval_steps=1, enemy_fire_cooldown_steps=None):
        super().__init__()

        if env_config is None:
            env_config = _load_sim_defaults()["env"]
        if reward_config is None:
            reward_config = _load_training_defaults()["reward"]

        self.render_mode = render_mode
        self.render_config = render_config  # lazily defaulted in render(); headless envs never touch this
        self.enemy_fire_enabled = bool(enemy_fire_enabled)
        # Curriculum knobs (see set_enemy_difficulty and train.curriculum in
        # config/training_default.yaml) — default to instant, full-cooldown enemy tracking,
        # i.e. exactly today's behavior, so nothing else that builds a CombatEnv needs to change.
        self.enemy_reaction_interval_steps = int(enemy_reaction_interval_steps)
        self._enemy_reaction_counter = 0

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

        # Enemy's own cooldown, independently curriculum-able — defaults to the soldier's
        # fire_cooldown_steps (today's behavior: both sides share one cooldown) unless overridden.
        self.enemy_fire_cooldown_steps = (
            int(enemy_fire_cooldown_steps) if enemy_fire_cooldown_steps is not None else self.fire_cooldown_steps
        )

        self.soldier_max_hp = float(env_config["soldier"]["max_hp"])
        self.enemy_max_hp = float(env_config["enemy"]["max_hp"])

        self.step_penalty = float(reward_config["step_penalty"])
        self.hit_reward_scale = float(reward_config["hit_reward_scale"])
        self.damage_taken_penalty_scale = float(reward_config["damage_taken_penalty_scale"])
        self.ammo_depleted_penalty = float(reward_config["ammo_depleted_penalty"])
        self.facing_enemy_reward = float(reward_config["facing_enemy_reward"])
        self.fire_while_facing_bonus = float(reward_config["fire_while_facing_bonus"])
        self.fire_while_not_facing_penalty = float(reward_config["fire_while_not_facing_penalty"])
        self.distance_closing_reward_scale = float(reward_config["distance_closing_reward_scale"])
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
        self.last_fire_events = []

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
        # Bypasses the reaction-interval gate: the enemy always starts an episode correctly
        # oriented (fair), the curriculum lag only kicks in from the first step() onward.
        self.enemy["orientation_bin"] = self._true_enemy_orientation_bin()
        self._enemy_reaction_counter = 0
        self.t = 0
        self.last_fire_events = []  # [{"shooter", "position", "orientation_bin", "facing_target",
                                     #   "hit", "target_position"}, ...] for the renderer

        obs = self._get_obs()
        info = {}
        return obs, info

    def step(self, action):
        move_dir, fire, orientation_bin = self._decode_action(action)

        self.soldier["cooldown_remaining"] = max(0, self.soldier["cooldown_remaining"] - 1)
        self.enemy["cooldown_remaining"] = max(0, self.enemy["cooldown_remaining"] - 1)

        prev_distance = geometry.distance(self.soldier["position"], self.enemy["position"])
        self.soldier["orientation_bin"] = orientation_bin
        self._move_soldier(move_dir)
        distance_closed = prev_distance - geometry.distance(self.soldier["position"], self.enemy["position"])
        self._update_enemy_orientation()

        # Computed here (post-move, pre-fire) since neither depends on fire resolution, and
        # both the reward and the fire-event log below need them.
        facing_enemy = self._soldier_facing_enemy()
        enemy_facing_soldier = self._enemy_facing_soldier()

        soldier_ammo_before = self.soldier["ammo"]
        soldier_hit, soldier_damage, soldier_ammo_consumed, soldier_fire_executed = (
            self._resolve_soldier_fire(fire)
        )
        ammo_just_depleted = soldier_ammo_before > 0 and self.soldier["ammo"] == 0
        enemy_hit, enemy_damage, enemy_ammo_consumed, enemy_fire_executed = self._resolve_enemy_fire()

        self.last_fire_events = []
        if soldier_fire_executed and soldier_ammo_consumed:
            self.last_fire_events.append({
                "shooter": "soldier",
                "position": tuple(self.soldier["position"]),
                "orientation_bin": self.soldier["orientation_bin"],
                "facing_target": facing_enemy,
                "hit": soldier_hit,
                "target_position": tuple(self.enemy["position"]) if soldier_hit else None,
            })
        if enemy_fire_executed and enemy_ammo_consumed:
            self.last_fire_events.append({
                "shooter": "enemy",
                "position": tuple(self.enemy["position"]),
                "orientation_bin": self.enemy["orientation_bin"],
                "facing_target": enemy_facing_soldier,
                "hit": enemy_hit,
                "target_position": tuple(self.soldier["position"]) if enemy_hit else None,
            })

        self.enemy["hp"] = max(0.0, self.enemy["hp"] - soldier_damage)
        self.soldier["hp"] = max(0.0, self.soldier["hp"] - enemy_damage)

        self.t += 1

        terminated, outcome = self._check_terminated()

        empty_gun_fire = soldier_fire_executed and not soldier_ammo_consumed
        real_fire_attempt = soldier_fire_executed and soldier_ammo_consumed
        fire_bearing_bin_distance = self._soldier_bearing_bin_distance() if real_fire_attempt else None
        reward = self._get_reward(
            soldier_damage_dealt=soldier_damage,
            enemy_damage_taken=enemy_damage,
            ammo_just_depleted=ammo_just_depleted,
            facing_enemy=facing_enemy,
            fire_bearing_bin_distance=fire_bearing_bin_distance,
            distance_closed=distance_closed,
            outcome=outcome,
        )

        obs = self._get_obs()
        info = {
            "outcome": outcome,
            "soldier_hit": soldier_hit,
            "enemy_hit": enemy_hit,
            "soldier_empty_gun_fire": empty_gun_fire,
        }
        truncated = False
        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode != "human":
            return None
        from rendering import pygame_renderer
        render_config = self.render_config
        if render_config is None:
            render_config = _load_sim_defaults()["render"]
        return pygame_renderer.render(self, render_config)

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

    def _true_enemy_orientation_bin(self):
        bearing = geometry.bearing(self.enemy["position"], self.soldier["position"])
        return geometry.angle_to_bin(bearing, self.bin_size_degrees)

    def _update_enemy_orientation(self):
        # Curriculum reaction lag: the enemy only re-acquires the soldier's true bearing once
        # every enemy_reaction_interval_steps ticks, holding its last orientation in between.
        # Since the soldier keeps moving, a stale orientation often fails the exact-bin
        # check_los match on its own — a sluggish enemy whiffs from bad tracking, not a
        # weakened accuracy roll. interval_steps=1 (default) updates every tick, i.e. today's
        # instant-tracking behavior, unchanged.
        self._enemy_reaction_counter += 1
        if self._enemy_reaction_counter < self.enemy_reaction_interval_steps:
            return
        self._enemy_reaction_counter = 0
        self.enemy["orientation_bin"] = self._true_enemy_orientation_bin()

    def set_enemy_difficulty(self, reaction_interval_steps=None, fire_cooldown_steps=None, fire_enabled=None):
        """Advance (or otherwise change) the curriculum mid-run without rebuilding the env —
        see train.curriculum in config/training_default.yaml and training/train.py."""
        if reaction_interval_steps is not None:
            self.enemy_reaction_interval_steps = int(reaction_interval_steps)
        if fire_cooldown_steps is not None:
            self.enemy_fire_cooldown_steps = int(fire_cooldown_steps)
        if fire_enabled is not None:
            self.enemy_fire_enabled = bool(fire_enabled)

    def _soldier_facing_enemy(self):
        bearing = geometry.bearing(self.soldier["position"], self.enemy["position"])
        true_bin = geometry.angle_to_bin(bearing, self.bin_size_degrees)
        return ballistics.check_los(self.soldier["orientation_bin"], true_bin)

    def _soldier_bearing_bin_distance(self):
        bearing = geometry.bearing(self.soldier["position"], self.enemy["position"])
        true_bin = geometry.angle_to_bin(bearing, self.bin_size_degrees)
        return geometry.bin_distance(self.soldier["orientation_bin"], true_bin, self.n_bins)

    def _enemy_facing_soldier(self):
        bearing = geometry.bearing(self.enemy["position"], self.soldier["position"])
        true_bin = geometry.angle_to_bin(bearing, self.bin_size_degrees)
        return ballistics.check_los(self.enemy["orientation_bin"], true_bin)

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
            self.enemy["cooldown_remaining"] = self.enemy_fire_cooldown_steps
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

    def _get_reward(self, soldier_damage_dealt, enemy_damage_taken, ammo_just_depleted, facing_enemy,
                     fire_bearing_bin_distance, distance_closed, outcome):
        reward = self.step_penalty
        reward += self.hit_reward_scale * soldier_damage_dealt
        reward -= self.damage_taken_penalty_scale * enemy_damage_taken
        if ammo_just_depleted:
            reward += self.ammo_depleted_penalty
        if facing_enemy:
            reward += self.facing_enemy_reward
        if fire_bearing_bin_distance is not None:
            # Linear interpolation over circular bin distance (0 = exact match, n_bins//2 =
            # dead opposite) instead of an all-or-nothing exact-match bonus/penalty. Endpoints
            # are unchanged (0 -> fire_while_facing_bonus, n_bins//2 -> fire_while_not_facing_
            # penalty) so existing exact-match/exact-opposite behavior is preserved; everything
            # in between now gives a graded signal instead of treating "1 bin off" the same as
            # "180 degrees off" -- see conversation/runs analysis: the exact-match-only version
            # gave zero gradient toward incrementally improving aim, which correlated with
            # trained policies collapsing to never firing at all rather than learning to aim.
            max_bin_distance = self.n_bins // 2
            frac = fire_bearing_bin_distance / max_bin_distance if max_bin_distance else 0.0
            reward += self.fire_while_facing_bonus * (1.0 - frac) + self.fire_while_not_facing_penalty * frac
        # Signed: positive when the soldier's move this tick reduced distance to the enemy,
        # negative when it increased it. Telescopes to scale * (initial_dist - final_dist)
        # over a whole episode, so oscillating back and forth nets ~0 — it can't be farmed,
        # only net progress toward (or away from) the enemy is rewarded.
        reward += self.distance_closing_reward_scale * distance_closed
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
