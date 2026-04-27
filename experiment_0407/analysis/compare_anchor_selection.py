import json
import math
import numpy as np
from scipy.optimize import least_squares

ANCHOR_POSITIONS = {
    1: (-0.75, 1.50),
    2: ( 0.75, 1.50),
    3: (-0.75, 0.00),
    4: ( 0.75, 0.00),
    5: (-0.57, -0.30),
    6: ( 0.57, -0.30),
}

C = 299702547
DWT_TIME_UNIT = 1 / (499.2e6 * 128.0)

def estimate_position_top2(record):
    root_anchor_id = record.get('root_anchor')
    messages = record.get('messages', {})
    adv = record.get('adv', {})

    if root_anchor_id is None:
        return None

    root_msg = messages.get(str(root_anchor_id))
    if not root_msg or len(root_msg) < 5:
        return None

    Ra, Da, Rb, Db, D2b = root_msg[:5]
    if any(ts < 0 for ts in (Ra, Da, Rb, Db, D2b)):
        return None

    numerator = Ra * Rb - Da * Db
    denominator = Ra + Rb + Da + Db
    tof_dtu = numerator / denominator if denominator != 0 else 0
    root_dist = tof_dtu * DWT_TIME_UNIT * C

    if root_anchor_id not in ANCHOR_POSITIONS:
        return None
    root_pos = ANCHOR_POSITIONS[root_anchor_id]

    leaf_candidates = []
    for anchor_id_str, msg in messages.items():
        anchor_id = int(anchor_id_str)
        if anchor_id == root_anchor_id:
            continue
        if not msg or len(msg) < 3:
            continue
        if msg[0] == 0 or msg[1] == 0 or msg[2] == 0:
            continue

        t2, t5, t8 = msg[0], msg[1], msg[2]
        UINT32 = 1 << 32
        Rt1 = (t5 - t2) % UINT32
        Rt2 = (t8 - t5) % UINT32

        denom = 2 * Rt1 + 2 * Rt2
        numer = Da * Rt1 - Rt2 * Ra + Rb * Rt1 - Rt2 * Db
        tdoa_dtu = numer / denom if denom != 0 else 0
        tdoa = tdoa_dtu * DWT_TIME_UNIT * C

        pos = ANCHOR_POSITIONS.get(anchor_id)
        if pos is None:
            continue

        power = adv.get(str(anchor_id), {}).get('power', 0)
        leaf_candidates.append((power, anchor_id, pos, tdoa))

    # power 높은 2개만
    leaf_candidates = sorted(leaf_candidates, key=lambda x: x[0], reverse=True)[:2]

    if not leaf_candidates:
        return None

    leaf_positions = [(pos, tdoa) for _, _, pos, tdoa in leaf_candidates]

    def residuals(p):
        x, y = p
        res = [np.linalg.norm([x - root_pos[0], y - root_pos[1]]) - root_dist]
        for (lx, ly), delta_d in leaf_positions:
            d_leaf = np.linalg.norm([x - lx, y - ly])
            d_leaf_root = np.linalg.norm([lx - root_pos[0], ly - root_pos[1]])
            res.append(d_leaf - d_leaf_root - delta_d)
        return res

    angles = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    candidates = [
        least_squares(residuals, x0=(
            root_pos[0] + root_dist * np.cos(a),
            root_pos[1] + root_dist * np.sin(a)
        )) for a in angles
    ]
    result = min((r for r in candidates if r.success), key=lambda r: r.cost, default=None)
    if result is None:
        return None

    return [round(result.x[0], 4), round(result.x[1], 4)]

def compare(filename):
    with open(filename) as f:
        records = json.load(f)

    full_errors = []
    top2_errors = []

    for r in records:
        pos = r.get('position')
        est_full = r.get('estimated_position')
        est_top2 = estimate_position_top2(r)

        if est_full and pos:
            full_errors.append(math.sqrt((est_full[0]-pos[0])**2 + (est_full[1]-pos[1])**2))

        if est_top2 and pos:
            top2_errors.append(math.sqrt((est_top2[0]-pos[0])**2 + (est_top2[1]-pos[1])**2))

    print(f"\n{'='*50}")
    print(f"파일: {filename}")
    print(f"\n  [전체 leaf - JSON 기존값]")
    if full_errors:
        print(f"    유효: {len(full_errors)}개")
        print(f"    평균 오차: {sum(full_errors)/len(full_errors):.4f} m")
        print(f"    최소 오차: {min(full_errors):.4f} m")
        print(f"    최대 오차: {max(full_errors):.4f} m")

    print(f"\n  [power 높은 leaf 2개만]")
    if top2_errors:
        print(f"    유효: {len(top2_errors)}개")
        print(f"    평균 오차: {sum(top2_errors)/len(top2_errors):.4f} m")
        print(f"    최소 오차: {min(top2_errors):.4f} m")
        print(f"    최대 오차: {max(top2_errors):.4f} m")

compare("log_20260407_231508_1.json")