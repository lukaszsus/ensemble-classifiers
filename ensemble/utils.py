import os

import numpy as np
from enum import Enum

from math import sqrt


def translate_class_labels(Y: np.ndarray):
    avail_y = list()
    for y in Y:
        if not y in avail_y:
            avail_y.append(y)
    avail_y.sort()
    return np.int_([avail_y.index(y) for y in Y])


class Distance(Enum):
    MANHATTAN  = 1
    EUCLIDEAN = 2


def count_one_over_sqrt_order_dist(distances):
    weights = list()
    for dist_for_one_point in distances:
        sorted = list(dist_for_one_point.copy())
        sorted.sort()  # Timsort - derived from merge sort and insertion sort
        weights.append([1/sqrt(sorted.index(dist) + 1) for dist in dist_for_one_point])
    return np.asarray(weights)


def create_dir_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)