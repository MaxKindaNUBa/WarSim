"""Pure math helpers shared by CombatEnv and its agents: euclidean distance,
compass bearing (0-360 degrees, atan2-based), and conversion between a
continuous bearing and a discrete orientation bin (half-open bins
[i*bin_size, (i+1)*bin_size), wrapping at 360). No env/agent state — every
function here takes plain positions/angles and is independently unit-tested
in tests/test_geometry.py.
"""
import math


def distance(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])


def bearing(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return math.degrees(math.atan2(dy, dx)) % 360.0


def angle_to_bin(angle, bin_size_degrees):
    n_bins = int(360 // bin_size_degrees)
    return int((angle % 360.0) // bin_size_degrees) % n_bins


def bin_to_angle_center(bin_idx, bin_size_degrees):
    return bin_idx * bin_size_degrees + bin_size_degrees / 2.0


def bin_distance(bin_a, bin_b, n_bins):
    """Circular distance between two orientation bins: 0 (exact match) to n_bins//2 (opposite)."""
    diff = abs(bin_a - bin_b) % n_bins
    return min(diff, n_bins - diff)
