import math
from itertools import combinations

import numpy as np

from settings import ANCHOR_POSITIONS
from utils import calc_azimuth, estimate_position_for_action
BETA = 5.0
GAMMA_DOP = 0.6

SUCCESS_RATES = {
    1: {1: 0.455, 2: 0.876, 3: 0.906, 4: 1.000, 5: 0.819, 6: 0.992},
    2: {1: 0.988, 2: 0.831, 3: 0.116, 4: 1.000, 5: 0.795, 6: 0.989},
    3: {1: 0.998, 2: 0.982, 3: 0.996, 4: 0.996, 5: 0.057, 6: 0.948},
    4: {1: 0.997, 2: 0.987, 3: 1.000, 4: 1.000, 5: 0.827, 6: 0.000},
    5: {1: 0.998, 2: 0.996, 3: 1.000, 4: 1.000, 5: 0.973, 6: 0.135},
    6: {1: 0.993, 2: 0.973, 3: 1.000, 4: 0.992, 5: 0.973, 6: 0.743},
    7: {1: 0.992, 2: 0.054, 3: 1.000, 4: 0.992, 5: 0.992, 6: 1.000},
}


def calc_gdop(tag_pos, anchor_ids):
    x, y = tag_pos
    H = []
    for aid in anchor_ids:
        ax, ay = ANCHOR_POSITIONS[aid]
        d = math.sqrt((x - ax) ** 2 + (y - ay) ** 2)
        if d == 0:
            continue
        H.append([(x - ax) / d, (y - ay) / d])

    H = np.array(H)
    if len(H) < 2:
        return 999.0

    try:
        HtH_inv = np.linalg.inv(H.T @ H)
        gdop = math.sqrt(np.trace(HtH_inv))
    except np.linalg.LinAlgError:
        gdop = 999.0

    return gdop



def calc_azimuth_error(est_pos, true_pos):
    est_az = calc_azimuth(est_pos[0], est_pos[1])
    true_az = calc_azimuth(true_pos[0], true_pos[1])
    err = abs(est_az - true_az)
    return 360 - err if err > 180 else err


def get_reward(record, primary, location):
    Rprimary = SUCCESS_RATES.get(location, {}).get(primary, 0.0)

    if record is None:
        return 0.0, {
            'case': 'primary_fail',
            'primary': primary,
            'Rprimary': 0.0,
        }

    return Rprimary, {
        'case': 'success',
        'primary': primary,
        'Rprimary': Rprimary,
    }