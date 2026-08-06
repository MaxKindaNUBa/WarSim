"""Randomized spawn logic for CombatEnv resets."""
import numpy as np


def spawn_positions(map_width, map_height, min_separation, rng):
    """Sample soldier/enemy positions within bounds, respecting min separation."""
    low = np.array([0.0, 0.0])
    high = np.array([map_width, map_height])
    while True:
        soldier_pos = rng.uniform(low, high)
        enemy_pos = rng.uniform(low, high)
        if np.hypot(*(enemy_pos - soldier_pos)) >= min_separation:
            return soldier_pos, enemy_pos


def spawn_orientation_bin(n_bins, rng):
    return int(rng.integers(0, n_bins))
