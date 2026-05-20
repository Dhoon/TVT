import os
import json
import numpy as np
from collections import defaultdict
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


def _gdop_from_H(H):
    try:
        HtH_inv = np.linalg.inv(H.T @ H)
        trace = np.trace(HtH_inv)
        return np.sqrt(trace) if trace > 0 else None
    except np.linalg.LinAlgError:
        return None


def compute_gdop(tag_pos, root_id, leaf_id):
    """ToF(root) + TDoA(leaf) 측정 모델 기반 GDOP"""
    tx, ty = tag_pos
    rx, ry = ANCHOR_POSITIONS[root_id]
    lx, ly = ANCHOR_POSITIONS[leaf_id]

    d_root = np.sqrt((tx - rx)**2 + (ty - ry)**2)
    d_leaf = np.sqrt((tx - lx)**2 + (ty - ly)**2)
    if d_root == 0 or d_leaf == 0:
        return None

    h_tof  = [(tx - rx) / d_root, (ty - ry) / d_root]
    h_tdoa = [(tx - lx) / d_leaf, (ty - ly) / d_leaf]
    return _gdop_from_H(np.array([h_tof, h_tdoa]))




def estimate_position(root_id, root_msg, leaf_id, messages, init_pos):
    Ra, Da, Rb, Db, D2b = root_msg[:5]
    if any(ts < 0 for ts in (Ra, Da, Rb, Db, D2b)):
        return None
    num = Ra * Rb - Da * Db
    den = Ra + Rb + Da + Db
    root_dist = (num / den if den != 0 else 0) * DWT_TIME_UNIT * C
    if root_dist < 0.5:
        return None

    root_pos = ANCHOR_POSITIONS[root_id]
    msg = messages.get(str(leaf_id))
    if not msg or len(msg) < 3:
        return None

    t2, t5, t8 = float(msg[0]), float(msg[1]), float(msg[2])
    if t2 == 0 or t5 == 0 or t8 == 0:
        return None

    UINT32 = 1 << 32
    Rt1 = (t5 - t2) % UINT32
    Rt2 = (t8 - t5) % UINT32
    denom = 2 * Rt1 + 2 * Rt2
    tdoa = (Da * Rt1 - Rt2 * Ra + Rb * Rt1 - Rt2 * Db) / (denom if denom else 1) * DWT_TIME_UNIT * C

    lx, ly = ANCHOR_POSITIONS[leaf_id]

    def residuals(p):
        x, y = p
        r0 = np.linalg.norm([x - root_pos[0], y - root_pos[1]]) - root_dist
        dl = np.linalg.norm([x - lx, y - ly])
        dlr = np.linalg.norm([lx - root_pos[0], ly - root_pos[1]])
        return [r0, dl - dlr - tdoa]

    result = least_squares(residuals, x0=init_pos)
    if not result.success:
        return None
    return result.x


def get_pedloc_leaf(root, adv):
    pl = [(int(k), v.get('power', 0)) for k, v in adv.items() if int(k) != root]
    pl.sort(key=lambda x: x[1], reverse=True)
    return pl[0][0] if pl else None


def collect_pairs(json_path, tag_pos, mode, scenario_name=None):
    """
    Returns list of dicts: {root, leaf, gdop, error}
    mode: 'pedloc' | 'davpl' | 'rr_best'
    """
    if not os.path.exists(json_path):
        return []

    with open(json_path, 'r', encoding='utf-8') as f:
        records = json.load(f)

    rows = []

    for r in records:
        root = r.get('root_anchor')
        if root is None:
            continue

        # init_pos = list(ANCHOR_POSITIONS[root])  # root anchor 위치로 cold start
        init_pos = tag_pos 

        messages = r.get('messages') or {}
        root_msg = messages.get(str(root))
        if not (root_msg and len(root_msg) >= 5 and all(v != 0 for v in root_msg[:5])):
            continue

        if mode == 'pedloc':
            adv = r.get('adv') or {}
            leaf = get_pedloc_leaf(root, adv)
            if leaf is None:
                continue
            if scenario_name == "Static 32.64m" and root == 2:
                continue
            est = estimate_position(root, root_msg, leaf, messages, init_pos)
            if est is None:
                continue
            chosen_leaf = leaf
            chosen_est = est

        elif mode == 'davpl':
            leaf_list = r.get('leaf') or []
            leaf = leaf_list[0] if leaf_list else None
            if leaf is None:
                continue
            if scenario_name == "Static 32.64m" and root == 2:
                continue
            est = estimate_position(root, root_msg, leaf, messages, init_pos)
            if est is None:
                continue
            chosen_leaf = leaf
            chosen_est = est

        elif mode == 'rr_best':
            best_est = None
            best_err = float('inf')
            best_leaf = None
            for leaf in range(1, 7):
                if leaf == root:
                    continue
                candidate = estimate_position(root, root_msg, leaf, messages, init_pos)
                if candidate is None:
                    continue
                err = np.linalg.norm([candidate[0] - tag_pos[0], candidate[1] - tag_pos[1]])
                if err > 50.0:
                    continue
                if err < best_err:
                    best_err = err
                    best_est = candidate
                    best_leaf = leaf
            if best_est is None:
                continue
            chosen_leaf = best_leaf
            chosen_est = best_est

        elif mode == 'rr_all':
            # 모든 leaf 조합을 각각 append (selection bias 없이 pair별 평균 오차 확인)
            for leaf in range(1, 7):
                if leaf == root:
                    continue
                candidate = estimate_position(root, root_msg, leaf, messages, init_pos)
                if candidate is None:
                    continue
                err = np.linalg.norm([candidate[0] - tag_pos[0], candidate[1] - tag_pos[1]])
                if err > 50.0:
                    continue
                gdop = compute_gdop(tag_pos, root, leaf)
                rows.append({
                    'root': root, 'leaf': leaf,
                    'gdop': gdop, 'error': err,
                    'est_x': candidate[0], 'est_y': candidate[1],
                })
            continue

        else:
            continue

        error = np.linalg.norm([chosen_est[0] - tag_pos[0], chosen_est[1] - tag_pos[1]])
        gdop = compute_gdop(tag_pos, root, chosen_leaf)
        rows.append({
            'root': root, 'leaf': chosen_leaf,
            'gdop': gdop, 'error': error,
            'est_x': chosen_est[0], 'est_y': chosen_est[1],
        })

    return rows


def print_pair_table(scenario_name, method, rows, tag_pos):
    if not rows:
        print(f"  {method}: no data")
        return

    # pair별 집계
    pair_data = defaultdict(lambda: {'gdops': [], 'errors': [], 'est_xs': [], 'est_ys': []})
    for row in rows:
        key = (row['root'], row['leaf'])
        if row['gdop'] is not None:
            pair_data[key]['gdops'].append(row['gdop'])
        pair_data[key]['errors'].append(row['error'])
        pair_data[key]['est_xs'].append(row['est_x'])
        pair_data[key]['est_ys'].append(row['est_y'])

    total = len(rows)
    mean_err_all = np.mean([r['error'] for r in rows])

    print(f"\n  [{method}]  total={total}  overall_mean_err={mean_err_all:.4f}m  tag={tag_pos}")
    print(f"  {'Pair':>10}  {'Count':>6}  {'Ratio':>7}  {'GDOP':>7}  {'MeanErr':>8}  {'StdErr':>7}  {'MeanX':>7}  {'MeanY':>7}")
    print(f"  {'-'*72}")

    for key in sorted(pair_data.keys(), key=lambda k: -len(pair_data[k]['errors'])):
        d = pair_data[key]
        cnt = len(d['errors'])
        ratio = cnt / total * 100
        gdop_str = f"{np.mean(d['gdops']):.2f}" if d['gdops'] else "  N/A"
        mean_err = np.mean(d['errors'])
        std_err  = np.std(d['errors'])
        mean_x   = np.mean(d['est_xs'])
        mean_y   = np.mean(d['est_ys'])
        pair_str = f"({key[0]},{key[1]})"
        print(f"  {pair_str:>10}  {cnt:>6}  {ratio:>6.1f}%  {gdop_str:>7}  {mean_err:>8.4f}  {std_err:>7.4f}  {mean_x:>7.3f}  {mean_y:>7.3f}")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))

    scenarios = {
        "Static 12.64m": {
            "RR-All":  (os.path.join(script_dir, "log_20260422_221958_mode1_10.json"), "rr_all"),
            "RR-Best": (os.path.join(script_dir, "log_20260422_221958_mode1_10.json"), "rr_best"),
            "PedLoc":  (os.path.join(script_dir, "log_20260422_224621_mode2_10.json"), "pedloc"),
            "DA-VPL":  (os.path.join(script_dir, "location_1_Static_12.64m.json"),    "davpl"),
        },
        "Static 22.64m": {
            "RR-All":  (os.path.join(script_dir, "log_20260422_221958_mode1_20.json"), "rr_all"),
            "RR-Best": (os.path.join(script_dir, "log_20260422_221958_mode1_20.json"), "rr_best"),
            "PedLoc":  (os.path.join(script_dir, "log_20260422_224621_mode2_20.json"), "pedloc"),
            "DA-VPL":  (os.path.join(script_dir, "location_2_Static_22.64m.json"),    "davpl"),
        },
        "Static 32.64m": {
            "RR-All":  (os.path.join(script_dir, "log_20260422_221958_mode1_30.json"), "rr_all"),
            "RR-Best": (os.path.join(script_dir, "log_20260422_221958_mode1_30.json"), "rr_best"),
            "PedLoc":  (os.path.join(script_dir, "log_20260422_224621_mode2_30.json"), "pedloc"),
            "DA-VPL":  (os.path.join(script_dir, "location_3_Static_32.64m.json"),    "davpl"),
        },
        "NLoS": {
            "RR-All":  (os.path.join(script_dir, "log_20260422_233919_mode1_nlos_1.json"), "rr_all"),
            "RR-Best": (os.path.join(script_dir, "log_20260422_233919_mode1_nlos_1.json"), "rr_best"),
            "PedLoc":  (os.path.join(script_dir, "log_20260422_234741_mode2_nlos_1.json"), "pedloc"),
            "DA-VPL":  (os.path.join(script_dir, "location_6_NLoS.json"),                 "davpl"),
        },
    }

    for scenario_name, paths in scenarios.items():
        tag_pos = TAG_POSITIONS[scenario_name]
        print(f"\n{'='*80}")
        print(f"Scenario: {scenario_name}  |  tag={tag_pos}")
        print(f"{'='*80}")

        for method, (path, mode) in paths.items():
            rows = collect_pairs(path, tag_pos, mode, scenario_name)
            print_pair_table(scenario_name, method, rows, tag_pos)
