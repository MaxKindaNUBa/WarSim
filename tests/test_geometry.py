import pytest

from envs import geometry


def test_distance_basic():
    assert geometry.distance((0, 0), (3, 4)) == pytest.approx(5.0)


def test_distance_zero():
    assert geometry.distance((5, 5), (5, 5)) == 0.0


@pytest.mark.parametrize("p1,p2,expected", [
    ((0, 0), (1, 0), 0.0),
    ((0, 0), (0, 1), 90.0),
    ((0, 0), (-1, 0), 180.0),
    ((0, 0), (0, -1), 270.0),
    ((0, 0), (1, 1), 45.0),
])
def test_bearing_cardinal(p1, p2, expected):
    assert geometry.bearing(p1, p2) == pytest.approx(expected)


def test_bearing_normalized_0_360():
    angle = geometry.bearing((0, 0), (0, -1))
    assert 0.0 <= angle < 360.0


def test_bearing_degenerate_zero_distance():
    assert geometry.bearing((5, 5), (5, 5)) == pytest.approx(0.0)


def test_angle_to_bin_zero():
    assert geometry.angle_to_bin(0.0, 10) == 0


def test_angle_to_bin_bin_edge_is_half_open():
    # bins are [i*bin_size, (i+1)*bin_size); the upper edge belongs to the next bin
    assert geometry.angle_to_bin(9.999, 10) == 0
    assert geometry.angle_to_bin(10.0, 10) == 1


def test_angle_to_bin_wraparound_359():
    assert geometry.angle_to_bin(359.0, 10) == 35


def test_angle_to_bin_wraparound_negative():
    # -1 degrees is equivalent to 359 degrees
    assert geometry.angle_to_bin(-1.0, 10) == 35


def test_angle_to_bin_wraparound_360_equals_zero():
    assert geometry.angle_to_bin(360.0, 10) == 0


def test_bin_to_angle_center():
    assert geometry.bin_to_angle_center(0, 10) == pytest.approx(5.0)
    assert geometry.bin_to_angle_center(35, 10) == pytest.approx(355.0)


def test_bin_roundtrip_center_maps_to_same_bin():
    bin_size = 10
    n_bins = 360 // bin_size
    for b in range(n_bins):
        center = geometry.bin_to_angle_center(b, bin_size)
        assert geometry.angle_to_bin(center, bin_size) == b
