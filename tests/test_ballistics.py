"""Unit tests for envs/ballistics.py: accuracy/damage table interpolation, the
exact-bin line-of-sight check, and resolve_shot()'s ammo/LOS/hit-roll gating.
"""
import numpy as np
import pytest

from envs import ballistics

ACCURACY_TABLE = {"distances": [0, 10, 20], "values": [1.0, 0.5, 0.0]}
DAMAGE_TABLE = {"distances": [0, 10, 20], "values": [40.0, 20.0, 0.0]}

TABLE_CONFIG = {
    "bin_size_degrees": 10,
    "accuracy_table": ACCURACY_TABLE,
    "damage_table": DAMAGE_TABLE,
}


def test_accuracy_lookup_interpolates():
    assert ballistics.accuracy_lookup(5, ACCURACY_TABLE) == pytest.approx(0.75)


def test_accuracy_lookup_exact_breakpoint():
    assert ballistics.accuracy_lookup(10, ACCURACY_TABLE) == pytest.approx(0.5)


def test_accuracy_lookup_clips_beyond_table():
    assert ballistics.accuracy_lookup(100, ACCURACY_TABLE) == pytest.approx(0.0)
    assert ballistics.accuracy_lookup(-5, ACCURACY_TABLE) == pytest.approx(1.0)


def test_damage_lookup_interpolates():
    assert ballistics.damage_lookup(5, DAMAGE_TABLE) == pytest.approx(30.0)


def test_check_los_true_when_bins_match():
    assert ballistics.check_los(3, 3) is True


def test_check_los_false_when_bins_differ():
    assert ballistics.check_los(3, 4) is False


def test_resolve_shot_no_ammo_never_hits_and_does_not_consume():
    shooter = {"position": (0, 0), "orientation_bin": 0, "ammo": 0}
    target = {"position": (10, 0)}
    hit, damage, ammo_consumed = ballistics.resolve_shot(shooter, target, TABLE_CONFIG)
    assert (hit, damage, ammo_consumed) == (False, 0.0, False)


def test_resolve_shot_no_los_consumes_ammo_but_never_hits():
    # target is due east of shooter (bearing 0 deg -> bin 0); shooter faces bin 18 (west) -> no LOS
    shooter = {"position": (0, 0), "orientation_bin": 18, "ammo": 5}
    target = {"position": (10, 0)}
    hit, damage, ammo_consumed = ballistics.resolve_shot(shooter, target, TABLE_CONFIG)
    assert hit is False
    assert damage == 0.0
    assert ammo_consumed is True


def test_resolve_shot_los_guaranteed_hit_at_zero_distance():
    # shooter and target coincide -> distance 0 -> accuracy 1.0 -> guaranteed hit
    shooter = {"position": (5, 5), "orientation_bin": 0, "ammo": 5}
    target = {"position": (5, 5)}
    rng = np.random.default_rng(0)
    hit, damage, ammo_consumed = ballistics.resolve_shot(shooter, target, TABLE_CONFIG, rng=rng)
    assert hit is True
    assert damage == pytest.approx(40.0)
    assert ammo_consumed is True


def test_resolve_shot_los_guaranteed_miss_beyond_table_range():
    shooter = {"position": (0, 0), "orientation_bin": 0, "ammo": 5}
    target = {"position": (100, 0)}
    rng = np.random.default_rng(0)
    hit, damage, ammo_consumed = ballistics.resolve_shot(shooter, target, TABLE_CONFIG, rng=rng)
    assert hit is False
    assert damage == 0.0
    assert ammo_consumed is True
