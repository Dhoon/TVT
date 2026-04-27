import math
from itertools import combinations

import numpy as np

from settings import ANCHOR_POSITIONS
from utils import calc_azimuth, estimate_position_for_action
BETA = 5.0
GAMMA_DOP = 0.6
LAMBDA_DOP = 1.0
LAMBDA_PEN = -2.0


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


def get_reward(record, action):
    primary, leaf1, leaf2 = action[0], action[1], action[2]
    true_pos = record.get('position')

    if true_pos is None:
        return -10.0

    est_pos = estimate_position_for_action(record, primary, leaf1, leaf2)
    if est_pos is None:
        return -10.0

    Et = calc_azimuth_error(est_pos, true_pos)
    Rangle = 1.0 / (1.0 + (Et / BETA) ** 2)

    remaining = [i for i in range(1, 7) if i != primary]
    best_error = float('inf')
    for l1, l2 in combinations(remaining, 2):
        pos = estimate_position_for_action(record, primary, l1, l2)
        if pos is None:
            continue
        err = calc_azimuth_error(pos, true_pos)
        if err < best_error:
            best_error = err

    if best_error == float('inf'):
        return -10.0

    Rbest = max(0, (best_error - BETA) / BETA)

    gdop = calc_gdop(est_pos, [primary, leaf1, leaf2])
    log_gdop = math.log10(gdop) if gdop > 0 else 0
    if log_gdop <= GAMMA_DOP:
        rdop = LAMBDA_DOP * (1 - log_gdop / GAMMA_DOP)
    else:
        rdop = LAMBDA_PEN * (log_gdop / GAMMA_DOP - 1)

    return Rangle - Rbest + rdop
