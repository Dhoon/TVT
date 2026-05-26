# Ablation: R_geom excluded from total reward (total = Rprimary + Rangle + Rbest)
import numpy as np

from settings import ANCHOR_POSITIONS
from utils import calc_azimuth, estimate_position_for_action, extract_ranging

BETA = 5.0

SUCCESS_RATES = {
    1: {1: 0.455, 2: 0.876, 3: 0.906, 4: 1.000, 5: 0.819, 6: 0.992},
    2: {1: 0.988, 2: 0.831, 3: 0.916, 4: 1.000, 5: 0.795, 6: 0.989},
    3: {1: 0.998, 2: 0.982, 3: 0.996, 4: 0.996, 5: 0.057, 6: 0.948},
    4: {1: 0.997, 2: 0.987, 3: 1.000, 4: 1.000, 5: 0.827, 6: 0.000},
    5: {1: 0.998, 2: 0.996, 3: 1.000, 4: 1.000, 5: 0.973, 6: 0.135},
    6: {1: 0.993, 2: 0.973, 3: 1.000, 4: 0.992, 5: 0.973, 6: 0.743},
    7: {1: 0.992, 2: 0.454, 3: 1.000, 4: 0.992, 5: 0.992, 6: 1.000},
}


def calc_geom(record, primary, leaf1):
    ranging = extract_ranging(record, primary, leaf1)
    if ranging is None:
        return 0.0
    root_dist, tdoa = ranging
    root_pos = np.array(ANCHOR_POSITIONS[primary])
    leaf_pos = np.array(ANCHOR_POSITIONS[leaf1])
    d_root_leaf = float(np.linalg.norm(leaf_pos - root_pos))
    if d_root_leaf == 0:
        return 0.0
    raw = (root_dist - d_root_leaf - tdoa) / d_root_leaf
    return float(np.clip(raw, -1.0, 1.0))


def calc_azimuth_error(est_pos, true_pos):
    from utils import calc_azimuth
    est_az  = calc_azimuth(est_pos[0], est_pos[1])
    true_az = calc_azimuth(true_pos[0], true_pos[1])
    err = abs(est_az - true_az)
    return 360 - err if err > 180 else err


def get_reward(record, action, location):
    primary, leaf1 = action[0], action[1]
    true_pos  = record.get('position')
    Rprimary  = SUCCESS_RATES.get(location, {}).get(primary, 0.0)
    R_geom    = calc_geom(record, primary, leaf1)  # computed for logging only

    if true_pos is None:
        return 0.0, {
            'case': 'no_true_pos',
            'Rprimary': 0.0, 'Rangle': 0.0, 'Rbest': 0.0, 'R_geom': R_geom,
        }

    est_pos, fail_type = estimate_position_for_action(record, primary, leaf1)

    if fail_type == 'primary':
        return -1.0, {
            'case': 'primary_fail',
            'Rprimary': 0.0, 'Rangle': 0.0, 'Rbest': 0.0, 'R_geom': R_geom,
        }

    remaining = [i for i in range(1, 7) if i != primary]
    best_error = float('inf')
    for l in remaining:
        pos, _ = estimate_position_for_action(record, primary, l)
        if pos is None:
            continue
        err = calc_azimuth_error(pos, true_pos)
        if err < best_error:
            best_error = err

    if est_pos is None:  # leaf fail
        if best_error == float('inf'):
            total = Rprimary - 1.0
            return total, {
                'case': 'leaf_no_best',
                'Rprimary': Rprimary, 'Rangle': 0.0, 'Rbest': 0.0, 'R_geom': R_geom,
            }
        Rbest = -(1.0 - best_error / BETA) if best_error <= BETA else 0.0
        total = Rprimary + Rbest
        return total, {
            'case': 'leaf_fail',
            'Rprimary': Rprimary, 'Rangle': 0.0, 'Rbest': Rbest, 'R_geom': R_geom,
        }

    angle_err = calc_azimuth_error(est_pos, true_pos)
    if angle_err <= BETA:
        Rangle = 1.0 - (angle_err / BETA)
        Rbest  = 0.0
    else:
        Rangle = max(-1.0, -(angle_err - BETA) / BETA)
        Rbest  = -(1.0 - best_error / BETA) if best_error <= BETA else 0.0

    total = Rprimary + Rangle + Rbest
    return total, {
        'case': 'success',
        'Rprimary': Rprimary, 'Rangle': Rangle, 'Rbest': Rbest, 'R_geom': R_geom,
    }
