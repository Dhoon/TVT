import os
import json
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

TAG_POSITIONS = {
    "Static 12.64m": (0.0,  12.64),
    "Static 22.64m": (0.0,  22.64),
    "Static 32.64m": (0.0,  32.64),
    "NLoS":          (-2.4, 22.64),
}

C = 299702547
DWT_TIME_UNIT = 1 / (499.2e6 * 128.0)


def estimate_position(root_anchor_id, root_msg, leaf_id, messages, init_pos):
    """
    primary anchor + leaf anchor 1개로 위치측위
    init_pos: 이전 측위값 (초기값)
    """
    Ra, Da, Rb, Db, D2b = root_msg[:5]
    if any(ts < 0 for ts in (Ra, Da, Rb, Db, D2b)):
        return None

    numerator   = Ra * Rb - Da * Db
    denominator = Ra + Rb + Da + Db
    tof_dtu     = numerator / denominator if denominator != 0 else 0
    root_dist   = tof_dtu * DWT_TIME_UNIT * C
    if root_dist < 0.5:
        return None

    root_pos = ANCHOR_POSITIONS[root_anchor_id]

    # leaf 1개 TDoA
    msg = messages.get(str(leaf_id))
    if not msg or len(msg) < 3:
        return None

    t2, t5, t8, t11 = msg[:4]
    if t2 == 0 or t5 == 0 or t8 == 0:
        return None

    UINT32 = 1 << 32
    Rt1 = (t5 - t2) % UINT32
    Rt2 = (t8 - t5) % UINT32

    denom    = 2 * Rt1 + 2 * Rt2
    numer    = Da * Rt1 - Rt2 * Ra + Rb * Rt1 - Rt2 * Db
    tdoa_dtu = numer / denom if denom != 0 else 0
    tdoa     = tdoa_dtu * DWT_TIME_UNIT * C

    leaf_pos = ANCHOR_POSITIONS.get(leaf_id)
    if leaf_pos is None:
        return None

    lx, ly = leaf_pos

    def residuals(p):
        x, y = p
        r0 = np.linalg.norm([x - root_pos[0], y - root_pos[1]]) - root_dist
        d_leaf      = np.linalg.norm([x - lx, y - ly]) #tag-leaf 거리
        d_leaf_root = np.linalg.norm([lx - root_pos[0], ly - root_pos[1]]) #root-leaf 거리
        r1 = d_leaf - d_leaf_root - tdoa
        return [r0, r1]

    result = least_squares(residuals, x0=init_pos)
    if not result.success:
        return None

    return result.x


def get_pedloc_leaf(root, adv):
    """PedLoc: adv power 기준 root 제외 1위 1개"""
    power_list = []
    for anchor_id_str, info in adv.items():
        anchor_id = int(anchor_id_str)
        if anchor_id == root:
            continue
        power_list.append((anchor_id, info.get('power', 0)))
    power_list.sort(key=lambda x: x[1], reverse=True)
    return power_list[0][0] if power_list else None


def analyze_localization(json_path, tag_pos, mode, scenario_name=None):
    if not os.path.exists(json_path):
        print(f"[SKIP] {json_path} 없음")
        return None

    with open(json_path, 'r', encoding='utf-8') as f:
        records = json.load(f)

    errors   = []
    prev_pos = [0.0, 0.0]  # 시나리오당 cold start 1회

    for r in records:
        root = r.get('root_anchor')
        if root is None:
            continue

        messages = r.get('messages') or {}
        root_msg = messages.get(str(root))

        if not (root_msg and len(root_msg) >= 5 and all(v != 0 for v in root_msg[:5])):
            continue

        if mode == 'pedloc':
            adv     = r.get('adv') or {}
            leaf_id = get_pedloc_leaf(root, adv)
            if leaf_id is None:
                continue
            if scenario_name == "Static 32.64m" and root == 2:
                continue
            est = estimate_position(root, root_msg, leaf_id, messages, prev_pos)

        elif mode == 'davpl':
            leaf_list = r.get('leaf') or []
            leaf_id   = leaf_list[0] if leaf_list else None
            if leaf_id is None:
                continue
            if scenario_name == "Static 32.64m" and root == 2:
                continue
            est = estimate_position(root, root_msg, leaf_id, messages, prev_pos)

        elif mode == 'rr_best':
            # 5개 leaf 모두 시도 → 오차 최소인 결과 선택 (50m 초과 발산 제거)
            best_est = None
            best_err = float('inf')
            for leaf_id in range(1, 7):
                if leaf_id == root:
                    continue
                candidate = estimate_position(root, root_msg, leaf_id, messages, prev_pos)
                if candidate is None:
                    continue
                err = np.linalg.norm([candidate[0] - tag_pos[0], candidate[1] - tag_pos[1]])
                if err > 50.0:
                    continue
                if err < best_err:
                    best_err = err
                    best_est = candidate
            est = best_est

        else:
            continue

        if est is None:
            continue

        prev_pos = list(est)
        error = np.linalg.norm([est[0] - tag_pos[0], est[1] - tag_pos[1]])
        errors.append(error)

    if not errors:
        return None

    return {
        "mean":   np.mean(errors),
        "std":    np.std(errors),
        "median": np.median(errors),
        "p95":    np.percentile(errors, 95),
        "count":  len(errors),
    }


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))

    scenarios = {
        "Static 12.64m": {
            "RR-Best":  (os.path.join(script_dir, "log_20260422_221958_mode1_10.json"),    "rr_best"),
            "PedLoc":   (os.path.join(script_dir, "log_20260422_224621_mode2_10.json"),    "pedloc"),
            "DA-VPL":   (os.path.join(script_dir, "location_1_Static_12.64m.json"),        "davpl"),
        },
        "Static 22.64m": {
            "RR-Best":  (os.path.join(script_dir, "log_20260422_221958_mode1_20.json"),    "rr_best"),
            "PedLoc":   (os.path.join(script_dir, "log_20260422_224621_mode2_20.json"),    "pedloc"),
            "DA-VPL":   (os.path.join(script_dir, "location_2_Static_22.64m.json"),        "davpl"),
        },
        "Static 32.64m": {
            "RR-Best":  (os.path.join(script_dir, "log_20260422_221958_mode1_30.json"),    "rr_best"),
            "PedLoc":   (os.path.join(script_dir, "log_20260422_224621_mode2_30.json"),    "pedloc"),
            "DA-VPL":   (os.path.join(script_dir, "location_3_Static_32.64m.json"),        "davpl"),
        },
        "NLoS": {
            "RR-Best":  (os.path.join(script_dir, "log_20260422_233919_mode1_nlos_1.json"),"rr_best"),
            "PedLoc":   (os.path.join(script_dir, "log_20260422_234741_mode2_nlos_1.json"),"pedloc"),
            "DA-VPL":   (os.path.join(script_dir, "location_6_NLoS.json"),                 "davpl"),
        },
    }

    print(f"\n{'='*75}")
    print(f"Localization Error Analysis (primary 1 + leaf 1)")
    print(f"  {'Scenario':<15} {'Method':<10} {'Count':>6} {'Mean':>8} {'Std':>8} {'Median':>8} {'P95':>8}")
    print(f"  {'-'*65}")

    for scenario_name, paths in scenarios.items():
        tag_pos = TAG_POSITIONS[scenario_name]
        first   = True
        for method, (path, mode) in paths.items():
            result = analyze_localization(path, tag_pos, mode, scenario_name)
            if result is None:
                print(f"  {scenario_name if first else '':<15} {method:<10} {'N/A':>6}")
                first = False
                continue
            print(
                f"  {scenario_name if first else '':<15} {method:<10} "
                f"{result['count']:>6} "
                f"{result['mean']:>8.4f} "
                f"{result['std']:>8.4f} "
                f"{result['median']:>8.4f} "
                f"{result['p95']:>8.4f}"
            )
            first = False
        print(f"  {'-'*65}")