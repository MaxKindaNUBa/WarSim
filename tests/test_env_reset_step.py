import numpy as np
import pytest

from envs import geometry
from envs.combat_env import CombatEnv


def _env_config(**overrides):
    cfg = {
        "map": {"width": 50.0, "height": 50.0},
        "bin_size_degrees": 10,
        "timestep_duration_s": 0.5,
        "episode": {"max_timesteps": 5, "min_spawn_separation": 5.0},
        "movement": {"step_size": 2.0},
        "gun": {
            "max_ammo": 2,
            "max_range": 100.0,
            "fire_cooldown_steps": 0,
            "accuracy_table": {"distances": [0, 100], "values": [1.0, 1.0]},
            "damage_table": {"distances": [0, 100], "values": [10.0, 10.0]},
        },
        "soldier": {"max_hp": 100},
        "enemy": {"max_hp": 100},
    }
    cfg.update(overrides)
    return cfg


def _reward_config():
    return {
        "hit_reward_scale": 1.0,
        "damage_taken_penalty_scale": 1.0,
        "step_penalty": -0.01,
        "terminal": {"win": 50.0, "lose": -50.0, "timeout": -10.0},
        "empty_gun_fire_penalty": -1.0,
        "orientation_shaping_bonus_weight": 0.05,
    }


def _make_env(enemy_fire_enabled=False, **env_overrides):
    return CombatEnv(
        env_config=_env_config(**env_overrides),
        reward_config=_reward_config(),
        enemy_fire_enabled=enemy_fire_enabled,
    )


def _fire_action_facing(env, orientation_bin):
    # move_dir=0 (stay), fire=1
    return 1 * env.n_bins + orientation_bin


def _orientation_bin_toward_enemy(env):
    bearing = geometry.bearing(env.soldier["position"], env.enemy["position"])
    return geometry.angle_to_bin(bearing, env.bin_size_degrees)


def test_reset_obs_shape_and_dtype():
    env = _make_env()
    obs, info = env.reset(seed=0)
    expected_dim = 2 + 4 + 4 + 1 + 1 + env.n_bins + env.n_bins
    assert obs.shape == (expected_dim,)
    assert obs.dtype == np.float32
    assert isinstance(info, dict)


def test_step_obs_shape_and_dtype_consistent_with_reset():
    env = _make_env()
    obs0, _ = env.reset(seed=0)
    obs1, reward, terminated, truncated, info = env.step(0)
    assert obs1.shape == obs0.shape
    assert obs1.dtype == obs0.dtype
    assert isinstance(reward, float)
    assert terminated in (True, False)
    assert truncated is False


def test_terminates_on_enemy_hp_depletion_outcome_win():
    env = _make_env(enemy_fire_enabled=False)
    env.reset(seed=1)
    orientation_bin = _orientation_bin_toward_enemy(env)
    action = _fire_action_facing(env, orientation_bin)

    obs, reward, terminated, truncated, info = env.step(action)

    assert env.enemy["hp"] == pytest.approx(90.0)
    assert terminated is False

    # a second guaranteed hit finishes the enemy off (100 -> 90 -> 80, keep firing)
    env.enemy["hp"] = 5.0
    obs, reward, terminated, truncated, info = env.step(action)
    assert terminated is True
    assert info["outcome"] == "win"


def test_terminates_on_soldier_hp_depletion_outcome_loss():
    env = _make_env(enemy_fire_enabled=True)
    env.reset(seed=2)
    env.soldier["hp"] = 5.0

    obs, reward, terminated, truncated, info = env.step(0)  # soldier does not fire

    assert terminated is True
    assert info["outcome"] == "loss"


def test_double_kill_counts_as_win():
    env = _make_env(enemy_fire_enabled=True)
    env.reset(seed=5)
    env.soldier["hp"] = 5.0
    env.enemy["hp"] = 5.0
    orientation_bin = _orientation_bin_toward_enemy(env)
    action = _fire_action_facing(env, orientation_bin)

    obs, reward, terminated, truncated, info = env.step(action)

    assert terminated is True
    assert info["outcome"] == "win"


def test_terminates_on_timeout():
    env = _make_env(episode={"max_timesteps": 2, "min_spawn_separation": 5.0})
    env.reset(seed=3)

    obs, reward, terminated, truncated, info = env.step(0)
    assert terminated is False

    obs, reward, terminated, truncated, info = env.step(0)
    assert terminated is True
    assert info["outcome"] == "timeout"


def test_ammo_decrements_on_every_fire_attempt_and_clamps_at_zero():
    env = _make_env(gun={
        "max_ammo": 1,
        "max_range": 100.0,
        "fire_cooldown_steps": 0,
        "accuracy_table": {"distances": [0, 100], "values": [0.0, 0.0]},
        "damage_table": {"distances": [0, 100], "values": [0.0, 0.0]},
    })
    env.reset(seed=4)
    orientation_bin = _orientation_bin_toward_enemy(env)
    action = _fire_action_facing(env, orientation_bin)

    assert env.soldier["ammo"] == 1

    obs, reward, terminated, truncated, info = env.step(action)
    assert env.soldier["ammo"] == 0  # the single round was consumed

    obs, reward, terminated, truncated, info = env.step(action)  # empty-gun fire attempt
    assert env.soldier["ammo"] == 0  # never goes negative
    # empty-gun penalty (-1.0) + step penalty (-0.01) + orientation bonus (+0.05), no hit/no damage
    assert reward == pytest.approx(-0.96)
