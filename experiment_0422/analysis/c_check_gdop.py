import os
import json
import numpy as np

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


def compute_gdop(tag_pos, anchor_ids):
    """primary 1 + leaf 1 = 2개 anchor 기준 GDOP"""
    tx, ty = tag_pos
    H = []
    for aid in anchor_ids:
        if aid not in ANCHOR_POSITIONS:
            return None
        ax, ay = ANCHOR_POSITIONS[aid]
        dist = np.sqrt((tx - ax)**2 + (ty - ay)**2)
        if dist == 0:
            return None
        H.append([(tx - ax) / dist, (ty - ay) / dist])
    H = np.array(H)
    try:
        HtH_inv = np.linalg.inv(H.T @ H)
        trace = np.trace(HtH_inv)
        if trace <= 0:
            return None
        return np.sqrt(trace)
    except np.linalg.LinAlgError:
        return None


def get_best_gdop(tag_pos, primary_anchor):
    """primary 고정 후 leaf 1개 선택 시 최소 GDOP"""
    candidates = [a for a in ANCHOR_POSITIONS.keys() if a != primary_anchor]
    best_gdop  = float('inf')
    best_leaf  = None
    for leaf in candidates:
        gdop = compute_gdop(tag_pos, [primary_anchor, leaf])
        if gdop is not None and gdop < best_gdop:
            best_gdop = gdop
            best_leaf = leaf
    return best_gdop, best_leaf


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


def analyze_gdop_pedloc(json_path, tag_pos):
    if not os.path.exists(json_path):
        print(f"[SKIP] {json_path} 없음")
        return None

    with open(json_path, 'r', encoding='utf-8') as f:
        records = json.load(f)

    actual_gdops = []
    best_gdops   = []

    for r in records:
        root = r.get('root_anchor')
        if root is None:
            continue

        messages = r.get('messages') or {}
        root_msg = messages.get(str(root))

        if not (root_msg and len(root_msg) >= 5 and all(v != 0 for v in root_msg[:5])):
            continue

        adv  = r.get('adv') or {}
        leaf = get_pedloc_leaf(root, adv)
        if leaf is None:
            continue

        actual_gdop = compute_gdop(tag_pos, [root, leaf])
        if actual_gdop is None:
            continue

        best_gdop, _ = get_best_gdop(tag_pos, root)
        actual_gdops.append(actual_gdop)
        best_gdops.append(best_gdop)

    if not actual_gdops:
        return None

    return {
        "actual_mean": np.mean(actual_gdops),
        "best_mean":   np.mean(best_gdops),
        "count":       len(actual_gdops),
    }


def analyze_gdop_davpl(json_path, tag_pos):
    if not os.path.exists(json_path):
        print(f"[SKIP] {json_path} 없음")
        return None

    with open(json_path, 'r', encoding='utf-8') as f:
        records = json.load(f)

    actual_gdops = []
    best_gdops   = []

    for r in records:
        root = r.get('root_anchor')
        if root is None:
            continue

        messages = r.get('messages') or {}
        root_msg = messages.get(str(root))

        if not (root_msg and len(root_msg) >= 5 and all(v != 0 for v in root_msg[:5])):
            continue

        leaf_list = r.get('leaf') or []
        if len(leaf_list) < 1:
            continue

        leaf = leaf_list[0]  # 첫 번째 leaf 1개만 사용

        actual_gdop = compute_gdop(tag_pos, [root, leaf])
        if actual_gdop is None:
            continue

        best_gdop, _ = get_best_gdop(tag_pos, root)
        actual_gdops.append(actual_gdop)
        best_gdops.append(best_gdop)

    if not actual_gdops:
        return None

    return {
        "actual_mean": np.mean(actual_gdops),
        "best_mean":   np.mean(best_gdops),
        "count":       len(actual_gdops),
    }


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))

    scenarios = {
        "Static 12.64m": {
            "PedLoc": (os.path.join(script_dir, "log_20260422_224621_mode2_10.json"), "pedloc"),
            "DA-VPL": (os.path.join(script_dir, "location_1_Static_12.64m.json"),    "davpl"),
        },
        "Static 22.64m": {
            "PedLoc": (os.path.join(script_dir, "log_20260422_224621_mode2_20.json"), "pedloc"),
            "DA-VPL": (os.path.join(script_dir, "location_2_Static_22.64m.json"),    "davpl"),
        },
        "Static 32.64m": {
            "PedLoc": (os.path.join(script_dir, "log_20260422_224621_mode2_30.json"), "pedloc"),
            "DA-VPL": (os.path.join(script_dir, "location_3_Static_32.64m.json"),    "davpl"),
        },
        "NLoS": {
            "PedLoc": (os.path.join(script_dir, "log_20260422_234741_mode2_nlos_1.json"), "pedloc"),
            "DA-VPL": (os.path.join(script_dir, "location_6_NLoS.json"),                 "davpl"),
        },
    }

    print(f"\n{'='*70}")
    print(f"GDOP Analysis: Actual vs Best (primary 1 + leaf 1)")
    print(f"  {'Scenario':<15} {'Method':<10} {'Count':>6} {'Actual GDOP':>12} {'Best GDOP':>10}")
    print(f"  {'-'*60}")

    for scenario_name, paths in scenarios.items():
        tag_pos = TAG_POSITIONS[scenario_name]
        first = True
        for method, (path, mode) in paths.items():
            if mode == "pedloc":
                result = analyze_gdop_pedloc(path, tag_pos)
            else:
                result = analyze_gdop_davpl(path, tag_pos)

            if result is None:
                print(f"  {scenario_name if first else '':<15} {method:<10} {'N/A':>6}")
                first = False
                continue

            print(
                f"  {scenario_name if first else '':<15} {method:<10} "
                f"{result['count']:>6} "
                f"{result['actual_mean']:>12.4f} "
                f"{result['best_mean']:>10.4f}"
            )
            first = False
        print(f"  {'-'*60}")