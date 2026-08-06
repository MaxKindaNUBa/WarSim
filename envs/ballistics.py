"""Accuracy/damage lookup tables, LOS gating, and shot resolution."""
import numpy as np

from . import geometry


def accuracy_lookup(distance, table):
    return float(np.interp(distance, table["distances"], table["values"]))


def damage_lookup(distance, table):
    return float(np.interp(distance, table["distances"], table["values"]))


def check_los(shooter_orientation_bin, true_bearing_bin):
    return shooter_orientation_bin == true_bearing_bin


def resolve_shot(shooter_state, target_state, table_config, rng=None):
    """Resolve a single fire attempt.

    shooter_state: {"position": (x, y), "orientation_bin": int, "ammo": int}
    target_state: {"position": (x, y)}
    table_config: {"bin_size_degrees": int, "accuracy_table": {...}, "damage_table": {...}}

    Returns (hit: bool, damage: float, ammo_consumed: bool).
    """
    if rng is None:
        rng = np.random.default_rng()

    ammo_consumed = shooter_state["ammo"] > 0
    if not ammo_consumed:
        return False, 0.0, False

    dist = geometry.distance(shooter_state["position"], target_state["position"])
    true_bearing = geometry.bearing(shooter_state["position"], target_state["position"])
    true_bearing_bin = geometry.angle_to_bin(true_bearing, table_config["bin_size_degrees"])

    if not check_los(shooter_state["orientation_bin"], true_bearing_bin):
        return False, 0.0, True

    accuracy = accuracy_lookup(dist, table_config["accuracy_table"])
    hit = bool(rng.random() < accuracy)
    damage = damage_lookup(dist, table_config["damage_table"]) if hit else 0.0

    return hit, damage, True
