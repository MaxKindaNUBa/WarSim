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
