"""Scripted heuristic baseline: always face the enemy's bearing, hold a
preferred standoff range, fire whenever the gun is ready (has ammo, off cooldown).
"""
from envs import geometry
from envs.combat_env import MOVE_VECTORS

# obs layout (envs/combat_env.py CombatEnv._get_obs): [dx, dy, soldier_ammo,
# soldier_max_range, soldier_cooldown_remaining, soldier_accuracy, ...]
_DX, _DY, _AMMO, _COOLDOWN = 0, 1, 2, 4


class HeuristicAgent:
    def __init__(self, n_bins, bin_size_degrees, preferred_range=15.0):
        self.n_bins = n_bins
        self.bin_size_degrees = bin_size_degrees
        self.preferred_range = preferred_range

    def act(self, obs):
        dx, dy = float(obs[_DX]), float(obs[_DY])
        ammo = obs[_AMMO]
        cooldown_remaining = obs[_COOLDOWN]

        distance = geometry.distance((0.0, 0.0), (dx, dy))
        bearing = geometry.bearing((0.0, 0.0), (dx, dy))
        orientation_bin = geometry.angle_to_bin(bearing, self.bin_size_degrees)

        move_dir = self._choose_move_dir(bearing, distance)
        fire = 1 if (ammo > 0 and cooldown_remaining <= 0) else 0

        return move_dir * (2 * self.n_bins) + fire * self.n_bins + orientation_bin

    def _choose_move_dir(self, heading, distance):
        if distance <= self.preferred_range:
            return 0  # hold position within the preferred standoff range

        best_dir, best_diff = 0, float("inf")
        for idx, (vx, vy) in enumerate(MOVE_VECTORS):
            if vx == 0.0 and vy == 0.0:
                continue
            vec_heading = geometry.bearing((0.0, 0.0), (vx, vy))
            diff = min(abs(vec_heading - heading), 360.0 - abs(vec_heading - heading))
            if diff < best_diff:
                best_diff, best_dir = diff, idx
        return best_dir
