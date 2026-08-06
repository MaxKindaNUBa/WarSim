"""Bearing/distance/binning math for the combat environment."""
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
