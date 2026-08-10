"""Circular replay buffers of (state, action, reward, next_state, done) transitions:
plain uniform-sample ReplayBuffer, and proportional-priority PrioritizedReplayBuffer.
"""
import numpy as np


class ReplayBuffer:
    def __init__(self, capacity, obs_dim):
        self.capacity = capacity
        self.states = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self._ptr = 0
        self._size = 0

    def push(self, state, action, reward, next_state, done):
        idx = self._ptr
        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_states[idx] = next_state
        self.dones[idx] = float(done)
        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def __len__(self):
        return self._size

    def sample(self, batch_size, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        idx = rng.choice(self._size, size=batch_size, replace=False)
        return (
            self.states[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_states[idx],
            self.dones[idx],
        )


class PrioritizedReplayBuffer:
    """Proportional-priority PER (Schaul et al.), same algorithm as the reference
    PrioritizedReplayBuffer in PA2_RLbonus.ipynb: P(i) = p_i^alpha / sum(p_j^alpha),
    importance-sampling weights w_i = (N * P(i))^-beta (normalized by max), priorities
    updated from |TD error| + epsilon after each train_step, new transitions pushed at
    max priority so they're sampled at least once.

    Differences from the notebook version: preallocated numpy arrays (circular buffer)
    instead of a deque, to match ReplayBuffer's style and avoid the O(n) `max(deque)`
    scan on every push; and beta is annealed on a start/end/decay_steps schedule (like
    DQNAgent.epsilon_at) driven by global_step, instead of a fixed per-sample-call
    increment, so it saturates to beta_end over roughly one full training run.

    This recomputes P(i) over the whole buffer on every sample() call (no sum-tree) —
    fine at this project's buffer scale (<=500k), same as the notebook's approach. A
    sum-tree would be the standard upgrade if capacity grows much larger.
    """

    def __init__(self, capacity, obs_dim, alpha=0.6, beta_start=0.4, beta_end=1.0,
                 beta_decay_steps=200_000, priority_epsilon=1e-6):
        self.capacity = capacity
        self.states = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.priorities = np.zeros(capacity, dtype=np.float64)
        self._ptr = 0
        self._size = 0

        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_decay_steps = beta_decay_steps
        self.priority_epsilon = priority_epsilon
        self._max_priority = 1.0  # new transitions start here, guaranteeing they get sampled at least once

    def push(self, state, action, reward, next_state, done):
        idx = self._ptr
        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_states[idx] = next_state
        self.dones[idx] = float(done)
        self.priorities[idx] = self._max_priority
        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def __len__(self):
        return self._size

    def beta_at(self, global_step):
        frac = min(1.0, global_step / max(1, self.beta_decay_steps))
        return self.beta_start + frac * (self.beta_end - self.beta_start)

    def sample(self, batch_size, global_step, rng=None):
        if rng is None:
            rng = np.random.default_rng()

        prios = self.priorities[:self._size]
        scaled = prios ** self.alpha
        probs = scaled / scaled.sum()

        indices = rng.choice(self._size, size=batch_size, p=probs)

        beta = self.beta_at(global_step)
        weights = (self._size * probs[indices]) ** (-beta)
        weights /= weights.max()  # normalize so the max IS weight is always 1 (stability, not correctness)

        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices],
            indices,
            weights.astype(np.float32),
        )

    def update_priorities(self, indices, td_errors):
        priorities = np.abs(td_errors) + self.priority_epsilon
        self.priorities[indices] = priorities
        self._max_priority = max(self._max_priority, float(priorities.max()))
