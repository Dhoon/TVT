import math

import numpy as np
from scipy.optimize import least_squares

from settings import ANCHOR_POSITIONS, C, DWT_TIME_UNIT


def calc_azimuth(est_x, est_y):
    az = math.degrees(math.atan2(est_x, est_y))
    return (az + 360) % 360


def extract_ranging(record, primary, leaf):
    """(root_dist, tdoa) 반환. 메시지 없으면 None."""
    messages = record.get('messages', {})

    root_msg = messages.get(str(primary))
    if not root_msg or len(root_msg) < 5:
        return None
    Ra, Da, Rb, Db, D2b = root_msg[:5]
    if any(ts < 0 for ts in (Ra, Da, Rb, Db, D2b)):
        return None
    num = Ra * Rb - Da * Db
    den = Ra + Rb + Da + Db
    root_dist = (num / den if den != 0 else 0) * DWT_TIME_UNIT * C

    msg = messages.get(str(leaf))
    if not msg or len(msg) < 3:
        return None
    t2, t5, t8 = float(msg[0]), float(msg[1]), float(msg[2])
    if t2 <= 0 or t5 <= 0 or t8 <= 0:
        return None
    UINT32 = 1 << 32
    Rt1 = (t5 - t2) % UINT32
    Rt2 = (t8 - t5) % UINT32
    denom = 2 * Rt1 + 2 * Rt2
    numer = Da * Rt1 - Rt2 * Ra + Rb * Rt1 - Rt2 * Db
    tdoa = (numer / denom if denom != 0 else 0) * DWT_TIME_UNIT * C

    return root_dist, tdoa


def estimate_position_for_action(record, primary, leaf1, prev_est=None):
    messages = record.get('messages', {})

    root_msg = messages.get(str(primary))
    if not root_msg or len(root_msg) < 5:
        return None, 'primary'

    Ra, Da, Rb, Db, D2b = root_msg[:5]
    if any(ts < 0 for ts in (Ra, Da, Rb, Db, D2b)):
        return None, 'primary'

    numerator = Ra * Rb - Da * Db
    denominator = Ra + Rb + Da + Db
    tof_dtu = numerator / denominator if denominator != 0 else 0
    root_dist = tof_dtu * DWT_TIME_UNIT * C

    root_pos = ANCHOR_POSITIONS[primary]
    leaf_positions = []
    tdoa_deltas = []

    for lid in [leaf1]:
        msg = messages.get(str(lid))
        if not msg or len(msg) < 3:
            continue
        t2, t5, t8 = msg[0], msg[1], msg[2]
        if t2 <= 0 or t5 <= 0 or t8 <= 0:
            continue

        UINT32 = 1 << 32
        Rt1 = (t5 - t2) % UINT32
        Rt2 = (t8 - t5) % UINT32

        denom = 2 * Rt1 + 2 * Rt2
        numer = Da * Rt1 - Rt2 * Ra + Rb * Rt1 - Rt2 * Db
        tdoa_dtu = numer / denom if denom != 0 else 0
        tdoa = tdoa_dtu * DWT_TIME_UNIT * C

        pos = ANCHOR_POSITIONS.get(lid)
        if pos:
            leaf_positions.append(pos)
            tdoa_deltas.append(tdoa)

    if not leaf_positions:
        return None, 'leaf'

    def residuals(p):
        x, y = p
        res = [np.linalg.norm([x - root_pos[0], y - root_pos[1]]) - root_dist]
        for (lx, ly), delta_d in zip(leaf_positions, tdoa_deltas):
            d_leaf = np.linalg.norm([x - lx, y - ly])
            d_leaf_root = np.linalg.norm([lx - root_pos[0], ly - root_pos[1]])
            res.append(d_leaf - d_leaf_root - delta_d)
        return res

    if prev_est is not None:
        x0 = prev_est
    else:
        x0 = root_pos  # cold start: root anchor 위치

    result = least_squares(residuals, x0=x0)
    if result.success:
        return result.x, None
    return None, 'leaf'
