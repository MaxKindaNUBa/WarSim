"""Unit tests for agents/dqn.py's DQNAgent: epsilon linear decay and the
curriculum epsilon-bump mechanic, and optimizer learning-rate decay
(disabled-by-default, linear-when-enabled, and applied live in train_step()).
"""
import numpy as np
import pytest

from agents.dqn import DQNAgent


def _make_agent(**overrides):
    kwargs = dict(
        obs_dim=4, n_actions=3, hidden_layers=[8],
        learning_rate=0.001, gamma=0.99, target_sync_interval_steps=100,
        epsilon_start=1.0, epsilon_end=0.01, epsilon_decay_steps=1000,
        seed=0,
    )
    kwargs.update(overrides)
    return DQNAgent(**kwargs)


def test_epsilon_at_linear_decay_default():
    agent = _make_agent()
    assert agent.epsilon_at(0) == pytest.approx(1.0)
    assert agent.epsilon_at(500) == pytest.approx(0.505)  # halfway: 1.0 + 0.5*(0.01-1.0)
    assert agent.epsilon_at(1000) == pytest.approx(0.01)
    assert agent.epsilon_at(2000) == pytest.approx(0.01)  # clamped past decay_steps


def test_bump_epsilon_resets_anchor_and_ramps_down_again():
    agent = _make_agent()
    assert agent.epsilon_at(1000) == pytest.approx(0.01)  # fully decayed

    agent.bump_epsilon(global_step=1000, value=0.5)
    assert agent.epsilon_at(1000) == pytest.approx(0.5)    # jumps immediately
    assert agent.epsilon_at(1500) == pytest.approx(0.255)  # halfway back down (same decay_steps=1000)
    assert agent.epsilon_at(2000) == pytest.approx(0.01)   # fully decayed again


def test_bump_epsilon_always_ramps_down_over_existing_decay_steps():
    agent = _make_agent()  # epsilon_decay_steps=1000, never overridden by a bump
    agent.bump_epsilon(global_step=1000, value=0.4)
    assert agent.epsilon_at(1000) == pytest.approx(0.4)
    assert agent.epsilon_at(1500) == pytest.approx(0.205)  # halfway over the original 1000-step window
    assert agent.epsilon_at(2000) == pytest.approx(0.01)   # fully decayed at the original pace


def test_bump_epsilon_defaults_value_to_epsilon_start():
    agent = _make_agent()
    agent.bump_epsilon(global_step=500)
    assert agent.epsilon_at(500) == pytest.approx(1.0)  # epsilon_start


def test_bump_epsilon_skipped_when_natural_epsilon_already_higher():
    agent = _make_agent()  # epsilon_start=1.0, epsilon_end=0.01, epsilon_decay_steps=1000
    natural = agent.epsilon_at(100)  # barely decayed yet
    assert natural > 0.35

    applied = agent.bump_epsilon(global_step=100, value=0.35)
    assert applied is False
    assert agent.epsilon_at(100) == pytest.approx(natural)  # unchanged -- bump must never lower epsilon


def test_bump_epsilon_returns_true_when_actually_applied():
    agent = _make_agent()
    applied = agent.bump_epsilon(global_step=1000, value=0.5)  # fully decayed (0.01) by step 1000
    assert applied is True
    assert agent.epsilon_at(1000) == pytest.approx(0.5)


def test_vdbe_requires_sigma():
    with pytest.raises(ValueError):
        _make_agent(epsilon_mode="vdbe")


def test_vdbe_epsilon_starts_at_epsilon_start():
    agent = _make_agent(epsilon_mode="vdbe", vdbe_sigma=1.0)
    assert agent.epsilon_at(0) == pytest.approx(1.0)  # ignores global_step entirely
    assert agent.epsilon_at(999_999) == pytest.approx(1.0)  # unchanged until a train_step happens


def test_vdbe_delta_defaults_to_inverse_n_actions():
    agent = _make_agent(epsilon_mode="vdbe", vdbe_sigma=1.0)  # n_actions=3
    assert agent.vdbe_delta == pytest.approx(1.0 / 3.0)


def test_update_vdbe_epsilon_matches_tokic_formula():
    # f = (1 - e^-x) / (1 + e^-x), x = |value_diff|/sigma; epsilon <- delta*f + (1-delta)*epsilon
    agent = _make_agent(epsilon_mode="vdbe", vdbe_sigma=2.0, vdbe_delta=0.5)
    agent._vdbe_epsilon = 0.2
    import math
    x = abs(3.0) / 2.0
    f = (1.0 - math.exp(-x)) / (1.0 + math.exp(-x))
    expected = 0.5 * f + 0.5 * 0.2
    result = agent._update_vdbe_epsilon(3.0)
    assert result == pytest.approx(expected)
    assert agent.epsilon_at(0) == pytest.approx(expected)


def test_update_vdbe_epsilon_zero_value_diff_pulls_toward_zero():
    # f(0) = 0 exactly, so epsilon decays toward 0 (not necessarily reaching it in one update)
    agent = _make_agent(epsilon_mode="vdbe", vdbe_sigma=1.0, vdbe_delta=0.5)
    agent._vdbe_epsilon = 0.8
    result = agent._update_vdbe_epsilon(0.0)
    assert result == pytest.approx(0.4)  # 0.5*0 + 0.5*0.8


def test_bump_epsilon_is_a_no_op_under_vdbe():
    agent = _make_agent(epsilon_mode="vdbe", vdbe_sigma=1.0)
    agent._vdbe_epsilon = 0.3
    applied = agent.bump_epsilon(global_step=1000, value=0.9)
    assert applied is False
    assert agent.epsilon_at(1000) == pytest.approx(0.3)  # unchanged, bump had no effect


def test_train_step_updates_vdbe_epsilon():
    agent = _make_agent(epsilon_mode="vdbe", vdbe_sigma=1.0, learning_rate=0.01)
    batch = (
        np.zeros((4, 4), dtype=np.float32),
        np.zeros(4, dtype=np.int64),
        np.ones(4, dtype=np.float32),  # nonzero reward -> nonzero TD-error -> epsilon should move
        np.zeros((4, 4), dtype=np.float32),
        np.zeros(4, dtype=np.float32),
    )
    before = agent.epsilon_at(0)
    agent.train_step(batch)
    after = agent.epsilon_at(0)
    assert after != pytest.approx(before)
    assert agent._last_vdbe_value_diff is not None


def test_train_step_does_not_touch_vdbe_state_in_linear_mode():
    agent = _make_agent()  # linear mode (default)
    batch = (
        np.zeros((4, 4), dtype=np.float32),
        np.zeros(4, dtype=np.int64),
        np.ones(4, dtype=np.float32),
        np.zeros((4, 4), dtype=np.float32),
        np.zeros(4, dtype=np.float32),
    )
    agent.train_step(batch)
    assert agent._last_vdbe_value_diff is None


def test_lr_at_disabled_by_default_stays_constant():
    agent = _make_agent(learning_rate=0.01)
    assert agent.lr_at(0) == pytest.approx(0.01)
    assert agent.lr_at(10_000) == pytest.approx(0.01)


def test_lr_at_linear_decay_when_enabled():
    agent = _make_agent(learning_rate=0.01, lr_decay_enabled=True, lr_end=0.001, lr_decay_steps=100)
    assert agent.lr_at(0) == pytest.approx(0.01)
    assert agent.lr_at(50) == pytest.approx(0.0055)  # halfway: 0.01 + 0.5*(0.001-0.01)
    assert agent.lr_at(100) == pytest.approx(0.001)
    assert agent.lr_at(200) == pytest.approx(0.001)  # clamped past decay_steps


def test_lr_decay_enabled_but_no_end_value_defaults_to_flat():
    agent = _make_agent(learning_rate=0.01, lr_decay_enabled=True, lr_decay_steps=100)
    assert agent.lr_at(0) == pytest.approx(0.01)
    assert agent.lr_at(50) == pytest.approx(0.01)  # lr_end defaults to lr_start -> no-op decay


def test_train_step_applies_decaying_lr_to_optimizer():
    agent = _make_agent(learning_rate=0.01, lr_decay_enabled=True, lr_end=0.001, lr_decay_steps=2)
    batch = (
        np.zeros((4, 4), dtype=np.float32),
        np.zeros(4, dtype=np.int64),
        np.zeros(4, dtype=np.float32),
        np.zeros((4, 4), dtype=np.float32),
        np.zeros(4, dtype=np.float32),
    )
    assert agent.optimizer.param_groups[0]["lr"] == pytest.approx(0.01)

    agent.train_step(batch)
    assert agent.optimizer.param_groups[0]["lr"] == pytest.approx(0.01)  # used lr_at(0) for this step

    agent.train_step(batch)
    assert agent.optimizer.param_groups[0]["lr"] == pytest.approx(0.0055)  # lr_at(1)

    agent.train_step(batch)
    assert agent.optimizer.param_groups[0]["lr"] == pytest.approx(0.001)  # lr_at(2), fully decayed


def test_train_step_leaves_lr_untouched_when_decay_disabled():
    agent = _make_agent(learning_rate=0.01)
    batch = (
        np.zeros((4, 4), dtype=np.float32),
        np.zeros(4, dtype=np.int64),
        np.zeros(4, dtype=np.float32),
        np.zeros((4, 4), dtype=np.float32),
        np.zeros(4, dtype=np.float32),
    )
    agent.train_step(batch)
    agent.train_step(batch)
    assert agent.optimizer.param_groups[0]["lr"] == pytest.approx(0.01)
