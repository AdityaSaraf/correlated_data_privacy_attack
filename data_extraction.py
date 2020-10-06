from math import exp
import numpy as np
import matplotlib.pyplot as plt
import math
from shapely.geometry import LineString, Point
import torch
import lstm_attack
import hmms
import argparse
import sys


def A(q, r, eps):
    er = exp(eps) * r
    root = math.sqrt((q - er) ** 2 + 4 * exp(eps) * (1 - q) * (1 - r))
    return abs(q - er) * root


def B(q, r, eps):
    return (q - exp(eps) * r) ** 2 + 2 * exp(eps) * (1 - q) * (1 - r)


def C(q, r, eps):
    return 2 * (1 - q) ** 2 * exp(eps)


# Represents A', the result of switching q and r in A
def A_alt(q, r, eps):
    return A(r, q, eps)


# Represents B', the result of switching q and r in B
def B_alt(q, r, eps):
    return B(r, q, eps)


# Represents C', the result of switching q and r in B
def C_alt(q, r, eps):
    return C(r, q, eps)


# Currently only returns the middle calculation, where neither rho_0 nor rho_1 are 0.5
# Returns (rho_0, rho_1) as described in Corollary 1
def min_exp_noise(q, r, eps):
    e = exp(eps)
    a = A(q, r, eps)
    b = B(q, r, eps)
    c = C(q, r, eps)
    a_ = A_alt(q, r, eps)
    b_ = B_alt(q, r, eps)
    c_ = C_alt(q, r, eps)
    if e >= q / r:
        line1 = LineString([(1 - (b + a) / (2 * c), 0.5), (0.5, c / (2 * (b + a)))])
    else:  # e < q/r
        line1 = LineString([(1 - (b - a) / (2 * c), 0.5), (0.5, c / (2 * (b - a)))])
    if e >= r / q:
        line2 = LineString(
            [(c_ / (2 * (b_ + a_)), 0.5), (0.5, 1 - (b_ + a_) / (2 * c_))]
        )
    else:  # e < r/q
        line2 = LineString(
            [(c_ / (2 * (b_ - a_)), 0.5), (0.5, 1 - (b_ - a_) / (2 * c_))]
        )

    (x_1, y_1), (x_2, y_2) = line1.coords
    slope1 = (y_2 - y_1) / (x_2 - x_1)
    (x_1, y_1), (x_2, y_2) = line2.coords
    slope2 = (y_2 - y_1) / (x_2 - x_1)
    line_top = LineString([(0, 0.5), (0.5, 0.5)])
    line_right = LineString([(0.5, 0), (0.5, 0.5)])
    if abs(slope2) > abs(slope1):
        intersect_top = line2.intersection(line_top)
        intersect_right = line1.intersection(line_right)
    else:
        intersect_top = line1.intersection(line_top)
        intersect_right = line2.intersection(line_right)
    intersect = line1.intersection(line2)
    pts = [intersect, intersect_top, intersect_right]
    exp_noise = np.zeros((3,))
    for i, pt in enumerate(pts):
        if isinstance(pt, Point):
            exp_noise[i] = pt.x * r / (q + r) + pt.y * q / (q + r)
        else:
            exp_noise[i] = float("inf")
    min_pt = pts[np.argmin(exp_noise)]
    return min_pt.x, min_pt.y
