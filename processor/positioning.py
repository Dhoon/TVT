import numpy as np
from scipy.optimize import least_squares

import state
from logger import log, timestamp
from settings import ANCHOR_POSITIONS


def estimate_tag_position(data):
    c = 299702547  # speed of light in m/s
    dwt_time_unit = 1 / (499.2e6 * 128.0)

    root_msg = next((m for m in data if m and m[0] == 23 and len(m) >= 11), None)
    if not root_msg:
        log(f"{timestamp()} [POSITION] No valid root anchor or TOF")
        return

    root_anchor_id = root_msg[1]
    tag_id = root_msg[2]
    sequence = root_msg[3]
    Ra, Da, Rb, Db, D2b = root_msg[6:11]

    numerator = Ra * Rb - Da * Db
    denominator = Ra + Rb + Da + Db
    tof_dtu = numerator / denominator if denominator != 0 else 0
    root_toa = tof_dtu * dwt_time_unit
    root_dist = root_toa * c
    log(f"{timestamp()} [TOF] Root TOF: {tof_dtu:.2f} DTU, Distance: {root_dist:.2f} m")

    if root_anchor_id not in ANCHOR_POSITIONS:
        log(f"{timestamp()} [POSITION] Unknown root anchor ID {root_anchor_id}")
        return
    root_pos = ANCHOR_POSITIONS[root_anchor_id]

    leaf_positions = []
    tdoa_deltas = []

    for msg in data:
        if not msg or msg[0] != 24 or len(msg) < 8:
            continue
        anchor_id = msg[1]
        t2, t5, t8, t11 = msg[4:8]

        Rt1 = t5 - t2
        Rt2 = t8 - t5
        if any(ts < 0 for ts in (Ra, Da, Rb, Db, D2b, Rt1, Rt2)):
            log(f"{timestamp()} [WARNING] Negative timestamp detected, skipping anchor {anchor_id}")
            continue

        denominator = 2 * Rt1 + 2 * Rt2
        numerator = Da * Rt1 - Rt2 * Ra + Rb * Rt1 - Rt2 * Db
        tdoa_dtu = numerator / denominator if denominator != 0 else 0
        tdoa = tdoa_dtu * dwt_time_unit * c

        pos = ANCHOR_POSITIONS.get(anchor_id)
        if pos:
            leaf_positions.append(pos)
            tdoa_deltas.append(tdoa)
        else:
            log(f"{timestamp()} [POSITION] Unknown leaf anchor ID {anchor_id}, skipping.")

    def residuals(p):
        x, y = p
        res = [np.linalg.norm([x - root_pos[0], y - root_pos[1]]) - root_dist]
        for (lx, ly), delta_d in zip(leaf_positions, tdoa_deltas):
            d_leaf = np.linalg.norm([x - lx, y - ly])
            d_leaf_root = np.linalg.norm([lx - root_pos[0], ly - root_pos[1]])
            res.append(d_leaf - d_leaf_root - delta_d)
        return res

    result = least_squares(residuals, x0=(0, 0))
    est_x, est_y = result.x
    if not result.success:
        log(f"{timestamp()} [POSITION] Optimization failed: {result.message}")
        return
    log(f"{timestamp()} [POSITION] Estimated tag position: ({est_x:.2f}, {est_y:.2f})")
    state.ui_queue.put({'type': 'position', 'x': est_x, 'y': est_y})
