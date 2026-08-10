"""Unit tests for agents/replay_buffer.py's PrioritizedReplayBuffer: circular-buffer
capacity wraparound, new-transition max-priority seeding, sample() shapes/weight
normalization, priority-proportional sampling bias, and beta annealing.
"""
import numpy as np
import pytest

from agents.replay_buffer import PrioritizedReplayBuffer


def _push_n(buf, n, obs_dim=3):
    for i in range(n):
        buf.push(
            state=np.full(obs_dim, i, dtype=np.float32),
            action=i % 4,
            reward=float(i),
            next_state=np.full(obs_dim, i + 1, dtype=np.float32),
            done=False,
        )


def test_len_and_capacity_wraparound():
    buf = PrioritizedReplayBuffer(capacity=5, obs_dim=3)
    assert len(buf) == 0
    _push_n(buf, 3)
    assert len(buf) == 3
    _push_n(buf, 10)  # push past capacity
    assert len(buf) == 5  # clamps, doesn't grow past capacity


def test_new_transitions_get_max_priority():
    buf = PrioritizedReplayBuffer(capacity=10, obs_dim=3)
    _push_n(buf, 3)
    assert (buf.priorities[:3] == 1.0).all()  # default max priority before any TD error seen

    buf.update_priorities(np.array([0]), np.array([50.0]))
    _push_n(buf, 1)  # next push should adopt the new (higher) max priority, not reset to 1.0
    assert buf.priorities[3] == pytest.approx(50.0 + buf.priority_epsilon)


def test_sample_shapes_and_index_validity():
    buf = PrioritizedReplayBuffer(capacity=20, obs_dim=3)
    _push_n(buf, 20)
    rng = np.random.default_rng(0)

    states, actions, rewards, next_states, dones, indices, weights = buf.sample(
        batch_size=8, global_step=0, rng=rng
    )

    assert states.shape == (8, 3)
    assert actions.shape == (8,)
    assert rewards.shape == (8,)
    assert next_states.shape == (8, 3)
    assert dones.shape == (8,)
    assert indices.shape == (8,)
    assert weights.shape == (8,)
    assert (indices >= 0).all() and (indices < 20).all()
    assert weights.max() == pytest.approx(1.0)  # normalized so the largest IS weight in the batch is 1
    assert (weights > 0).all()


def test_higher_priority_transitions_sampled_more_often():
    buf = PrioritizedReplayBuffer(capacity=10, obs_dim=1, alpha=1.0)
    _push_n(buf, 10, obs_dim=1)
    # crank index 0's priority way above everything else
    buf.update_priorities(np.array([0]), np.array([1000.0]))
    for i in range(1, 10):
        buf.update_priorities(np.array([i]), np.array([0.0]))

    rng = np.random.default_rng(0)
    _, _, _, _, _, indices, _ = buf.sample(batch_size=2000, global_step=0, rng=rng)
    frac_index_0 = (indices == 0).mean()
    assert frac_index_0 > 0.9  # should dominate the sample given its enormous relative priority


def test_beta_at_anneals_linearly_and_clamps_at_end():
    buf = PrioritizedReplayBuffer(capacity=10, obs_dim=1, beta_start=0.4, beta_end=1.0, beta_decay_steps=100)
    assert buf.beta_at(0) == pytest.approx(0.4)
    assert buf.beta_at(50) == pytest.approx(0.7)
    assert buf.beta_at(100) == pytest.approx(1.0)
    assert buf.beta_at(1000) == pytest.approx(1.0)  # clamped, doesn't overshoot past beta_end
