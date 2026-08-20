"""Integration tests for envs/combat_env.py: reset()/step() observation shape,
episode termination (win/loss/timeout/double-kill), ammo depletion, per-tick
reward-term arithmetic, the enemy reaction-lag curriculum mechanic, and
set_enemy_difficulty()'s live-update behavior.
"""
import numpy as np
import pytest

from envs import geometry
from envs.combat_env import MOVE_VECTORS, CombatEnv


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


def _reward_config(**overrides):
    cfg = {
        "hit_reward_scale": 1.0,
        "damage_taken_penalty_scale": 1.0,
        "step_penalty": -0.01,
        "terminal": {"win": 50.0, "lose": -50.0, "timeout": -10.0},
        "ammo_depleted_penalty": -1.0,
        # 0.0 by default so it's a no-op for tests that don't care about it (most of the
        # reward-arithmetic assertions below predate this reward term); see
        # test_facing_enemy_reward_* for dedicated coverage with a nonzero value.
        "facing_enemy_reward": 0.0,
        "fire_while_facing_bonus": 0.05,
        "fire_while_not_facing_penalty": -0.05,
        "distance_closing_reward_scale": 0.1,
    }
    cfg.update(overrides)
    return cfg


def _make_env(enemy_fire_enabled=False, reward_overrides=None,
               enemy_reaction_interval_steps=1, enemy_fire_cooldown_steps=None, **env_overrides):
    return CombatEnv(
        env_config=_env_config(**env_overrides),
        reward_config=_reward_config(**(reward_overrides or {})),
        enemy_fire_enabled=enemy_fire_enabled,
        enemy_reaction_interval_steps=enemy_reaction_interval_steps,
        enemy_fire_cooldown_steps=enemy_fire_cooldown_steps,
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
    # ammo_depleted_penalty (-1.0) is paid exactly on the tick ammo hits 0, on top of the
    # usual fire-while-facing bonus (+0.05) + step penalty (-0.01); guaranteed miss, no hit reward
    assert reward == pytest.approx(-0.01 + 0.05 - 1.0)

    obs, reward, terminated, truncated, info = env.step(action)  # empty-gun fire attempt
    assert env.soldier["ammo"] == 0  # never goes negative
    # ammo was already 0 before this attempt, so ammo_depleted_penalty does NOT fire again —
    # just the step penalty (no fire-while-facing bonus either: no round was actually fired)
    assert reward == pytest.approx(-0.01)


def test_reward_bonus_for_firing_while_facing_enemy():
    env = _make_env(gun={
        "max_ammo": 5, "max_range": 100.0, "fire_cooldown_steps": 0,
        "accuracy_table": {"distances": [0, 100], "values": [0.0, 0.0]},  # guaranteed miss
        "damage_table": {"distances": [0, 100], "values": [0.0, 0.0]},
    })
    env.reset(seed=6)
    orientation_bin = _orientation_bin_toward_enemy(env)
    action = _fire_action_facing(env, orientation_bin)

    obs, reward, terminated, truncated, info = env.step(action)
    # step penalty (-0.01) + fire-while-facing bonus (+0.05); guaranteed miss, so no hit reward
    assert reward == pytest.approx(-0.01 + 0.05)


def test_reward_penalty_for_firing_while_not_facing_enemy():
    env = _make_env(gun={
        "max_ammo": 5, "max_range": 100.0, "fire_cooldown_steps": 0,
        "accuracy_table": {"distances": [0, 100], "values": [0.0, 0.0]},
        "damage_table": {"distances": [0, 100], "values": [0.0, 0.0]},
    })
    env.reset(seed=6)
    true_bin = _orientation_bin_toward_enemy(env)
    wrong_bin = (true_bin + env.n_bins // 2) % env.n_bins  # 180 degrees off, guaranteed no LOS
    action = 1 * env.n_bins + wrong_bin  # move_dir=0, fire=1, wrong orientation

    obs, reward, terminated, truncated, info = env.step(action)
    # step penalty (-0.01) + fire-while-not-facing penalty (-0.05); no LOS, so no hit roll at all
    assert reward == pytest.approx(-0.01 - 0.05)


def test_reward_for_firing_near_but_not_exactly_facing_enemy_is_graded():
    env = _make_env(gun={
        "max_ammo": 5, "max_range": 100.0, "fire_cooldown_steps": 0,
        "accuracy_table": {"distances": [0, 100], "values": [0.0, 0.0]},
        "damage_table": {"distances": [0, 100], "values": [0.0, 0.0]},
    })
    env.reset(seed=6)
    true_bin = _orientation_bin_toward_enemy(env)
    near_bin = (true_bin + 1) % env.n_bins  # one bin off, not an exact LOS match
    action = 1 * env.n_bins + near_bin  # move_dir=0, fire=1

    obs, reward, terminated, truncated, info = env.step(action)

    # linear interpolation over circular bin distance (bin_distance=1 of max n_bins//2=18):
    # frac = 1/18 -> reward = 0.5*(1 - 1/18) + (-0.5)*(1/18), plus the flat step penalty
    max_bin_distance = env.n_bins // 2
    frac = 1 / max_bin_distance
    expected_bearing_term = 0.05 * (1.0 - frac) + (-0.05) * frac
    assert reward == pytest.approx(-0.01 + expected_bearing_term)
    # strictly better than firing dead-opposite (-0.05 full penalty), strictly worse than an
    # exact match (+0.05 full bonus) -- a real gradient, not a cliff
    assert -0.01 - 0.05 < reward < -0.01 + 0.05


def test_facing_enemy_reward_applies_every_tick_even_without_firing():
    env = _make_env(reward_overrides={"facing_enemy_reward": 0.005})
    env.reset(seed=6)
    orientation_bin = _orientation_bin_toward_enemy(env)
    action = 0 * (2 * env.n_bins) + 0 * env.n_bins + orientation_bin  # stay, no fire, face enemy

    obs, reward, terminated, truncated, info = env.step(action)
    # step penalty (-0.01) + facing_enemy_reward (+0.005); no fire attempted at all
    assert reward == pytest.approx(-0.01 + 0.005)


def test_facing_enemy_reward_withheld_when_not_facing():
    env = _make_env(reward_overrides={"facing_enemy_reward": 0.005})
    env.reset(seed=6)
    true_bin = _orientation_bin_toward_enemy(env)
    wrong_bin = (true_bin + env.n_bins // 2) % env.n_bins
    action = 0 * (2 * env.n_bins) + 0 * env.n_bins + wrong_bin  # stay, no fire, face away

    obs, reward, terminated, truncated, info = env.step(action)
    # just the step penalty — not facing the enemy, so no dense reward this tick
    assert reward == pytest.approx(-0.01)


def test_enemy_reaction_interval_one_updates_every_call():
    env = _make_env(enemy_reaction_interval_steps=1)
    env.reset(seed=10)
    env.soldier["position"] = np.array([0.0, 10.0])
    env._update_enemy_orientation()
    assert env.enemy["orientation_bin"] == env._true_enemy_orientation_bin()


def test_enemy_reaction_lag_only_updates_orientation_every_n_calls():
    env = _make_env(enemy_reaction_interval_steps=3)
    env.reset(seed=10)
    env.soldier["position"] = np.array([0.0, 0.0])
    env.enemy["position"] = np.array([10.0, 0.0])
    env.enemy["orientation_bin"] = env._true_enemy_orientation_bin()
    stale_bin = env.enemy["orientation_bin"]

    # Move the soldier to a position with a very different true bearing from the enemy,
    # guaranteeing the orientation bin would change if the enemy re-acquired it now.
    env.soldier["position"] = np.array([0.0, 10.0])
    assert env._true_enemy_orientation_bin() != stale_bin  # sanity: the true bearing really did change

    env._update_enemy_orientation()
    assert env.enemy["orientation_bin"] == stale_bin  # call 1/3: still stale
    env._update_enemy_orientation()
    assert env.enemy["orientation_bin"] == stale_bin  # call 2/3: still stale
    env._update_enemy_orientation()
    assert env.enemy["orientation_bin"] == env._true_enemy_orientation_bin()  # call 3/3: refreshed


def test_enemy_fire_cooldown_defaults_to_shared_fire_cooldown_steps():
    env = _make_env(gun={
        "max_ammo": 5, "max_range": 100.0, "fire_cooldown_steps": 7,
        "accuracy_table": {"distances": [0, 100], "values": [1.0, 1.0]},
        "damage_table": {"distances": [0, 100], "values": [10.0, 10.0]},
    })
    assert env.enemy_fire_cooldown_steps == 7  # not overridden -> matches the soldier's cooldown


def test_enemy_fire_cooldown_can_be_overridden_independently_of_soldier():
    env = _make_env(enemy_fire_cooldown_steps=20, gun={
        "max_ammo": 5, "max_range": 100.0, "fire_cooldown_steps": 3,
        "accuracy_table": {"distances": [0, 100], "values": [1.0, 1.0]},
        "damage_table": {"distances": [0, 100], "values": [10.0, 10.0]},
    })
    assert env.fire_cooldown_steps == 3        # soldier's own cooldown, unaffected
    assert env.enemy_fire_cooldown_steps == 20


def test_set_enemy_difficulty_updates_both_knobs_live():
    env = _make_env()
    assert env.enemy_reaction_interval_steps == 1

    env.set_enemy_difficulty(reaction_interval_steps=8, fire_cooldown_steps=15)
    assert env.enemy_reaction_interval_steps == 8
    assert env.enemy_fire_cooldown_steps == 15

    env.set_enemy_difficulty(reaction_interval_steps=2)  # fire_cooldown_steps omitted -> unchanged
    assert env.enemy_reaction_interval_steps == 2
    assert env.enemy_fire_cooldown_steps == 15


def test_set_enemy_difficulty_can_toggle_fire_enabled():
    env = _make_env(enemy_fire_enabled=True)
    assert env.enemy_fire_enabled is True

    env.set_enemy_difficulty(fire_enabled=False)
    assert env.enemy_fire_enabled is False
    # other knobs omitted -> unchanged
    assert env.enemy_reaction_interval_steps == 1

    env.set_enemy_difficulty(fire_enabled=True)
    assert env.enemy_fire_enabled is True


def test_reward_bonus_for_closing_distance_to_enemy():
    env = _make_env()
    env.reset(seed=7)
    env.soldier["position"] = np.array([0.0, 0.0])
    env.enemy["position"] = np.array([10.0, 0.0])
    move_dir = MOVE_VECTORS.index((1.0, 0.0))  # east, i.e. toward the enemy
    action = move_dir * (2 * env.n_bins)  # fire=0, orientation_bin=0 (irrelevant, not firing)

    obs, reward, terminated, truncated, info = env.step(action)
    # soldier moves 2.0 units toward the enemy (step_size=2.0): distance drops 10 -> 8
    # step penalty (-0.01) + distance_closing_reward_scale (0.1) * distance_closed (+2.0)
    assert reward == pytest.approx(-0.01 + 0.1 * 2.0)


def test_reward_penalty_for_increasing_distance_to_enemy():
    env = _make_env()
    env.reset(seed=7)
    env.soldier["position"] = np.array([20.0, 0.0])
    env.enemy["position"] = np.array([30.0, 0.0])
    move_dir = MOVE_VECTORS.index((-1.0, 0.0))  # west, i.e. away from the enemy
    action = move_dir * (2 * env.n_bins)

    obs, reward, terminated, truncated, info = env.step(action)
    # soldier moves 2.0 units away from the enemy: distance grows 10 -> 12
    # step penalty (-0.01) + distance_closing_reward_scale (0.1) * distance_closed (-2.0)
    assert reward == pytest.approx(-0.01 - 0.1 * 2.0)
