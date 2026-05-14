import os
import json
from collections import defaultdict


def get_anchor_distribution(json_path, missing_anchor=None):
    """
    주어진 json 파일에서 root_anchor 선택 분포를 계산합니다.

    missing_anchor: int or None
        None이면 실 데이터 그대로 사용
        int(예: 6)이면 해당 anchor가 누락된 것으로 보정:
          - 보정 total = 실제 total * 6/5
          - anchor 1~5: count / 보정 total * 100
          - missing anchor: (보정 total - 실제 total) / 보정 total * 100
    """
    if not os.path.exists(json_path):
        print(f"[SKIP] {json_path} 없음")
        return None

    with open(json_path, 'r', encoding='utf-8') as f:
        records = json.load(f)

    anchor_count = defaultdict(int)
    total = 0

    for r in records:
        root = r.get('root_anchor')
        if root is None:
            continue
        anchor_count[root] += 1
        total += 1

    if total == 0:
        return total, {}

    if missing_anchor is not None:
        corrected_total = total * 6 / 5
        distribution = {anchor: count / corrected_total * 100 for anchor, count in anchor_count.items()}
        distribution[missing_anchor] = (corrected_total - total) / corrected_total * 100
    else:
        distribution = {anchor: count / total * 100 for anchor, count in anchor_count.items()}

    return total, distribution


def print_distribution_table(scenario_name, paths: dict, all_anchors=None):
    """
    paths: {'Method': (json_path, missing_anchor)}
    """
    print(f"\n{'='*70}")
    print(f"Scenario: {scenario_name}")

    if all_anchors is None:
        all_anchors = list(range(1, 7))  # Anchor 1~6

    header = f"  {'Method':<15} {'Total':>6}" + "".join(f"  A{a:>2}" for a in all_anchors)
    print(header)
    print(f"  {'-'*70}")

    results = {}
    for method, (path, missing_anchor) in paths.items():
        result = get_anchor_distribution(path, missing_anchor=missing_anchor)
        if result is None:
            continue
        total, dist = result

        row = f"  {method:<15} {total:>6}"
        for a in all_anchors:
            val = dist.get(a, 0.0)
            row += f"  {val:>4.1f}"
        print(row)
        results[method] = dist

    return results


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # paths: {'Method': (json_path, missing_anchor)}
    # Round-Robin: missing_anchor=6 (로그 버그 보정)
    # PedLoc: missing_anchor=None (실 데이터 그대로)
    # DA-VPL: missing_anchor=None (실 데이터 그대로)
    scenarios = {
        "Static 12.64m": {
            "Round-Robin": (os.path.join(script_dir, "log_20260422_221958_mode1_10.json"), 6),
            "PedLoc":      (os.path.join(script_dir, "log_20260422_224621_mode2_10.json"), None),
            "DA-VPL":      (os.path.join(script_dir, "location_1_Static_12.64m.json"), None),
        },
        "Static 22.64m": {
            "Round-Robin": (os.path.join(script_dir, "log_20260422_221958_mode1_20.json"), 6),
            "PedLoc":      (os.path.join(script_dir, "log_20260422_224621_mode2_20.json"), None),
            "DA-VPL":      (os.path.join(script_dir, "location_2_Static_22.64m.json"), None),
        },
        "Static 32.64m": {
            "Round-Robin": (os.path.join(script_dir, "log_20260422_221958_mode1_30.json"), 6),
            "PedLoc":      (os.path.join(script_dir, "log_20260422_224621_mode2_30.json"), None),
            "DA-VPL":      (os.path.join(script_dir, "location_3_Static_32.64m.json"), None),
        },
        "Moving (22.64m->2.64m)": {
            "Round-Robin": (os.path.join(script_dir, "log_20260422_221958_mode1_20-0.json"), 6),
            "PedLoc":      (os.path.join(script_dir, "log_20260422_224621_mode2_20-0.json"), None),
            "DA-VPL":      (os.path.join(script_dir, "location_4_Moving_(22.64m-to2.64m).json"), None),
        },
        "Moving (2.64m->22.64m)": {
            "Round-Robin": (os.path.join(script_dir, "log_20260422_221958_mode1_0-20.json"), 6),
            "PedLoc":      (os.path.join(script_dir, "log_20260422_224621_mode2_0-20.json"), None),
            "DA-VPL":      (os.path.join(script_dir, "location_5_Moving_(2.64m-to22.64m).json"), None),
        },
        "NLoS": {
            "Round-Robin": (os.path.join(script_dir, "log_20260422_233919_mode1_nlos_1.json"), 6),
            "PedLoc":      (os.path.join(script_dir, "log_20260422_234741_mode2_nlos_1.json"), None),
            "DA-VPL":      (os.path.join(script_dir, "location_6_NLoS.json"), None),
        },
    }

    for scenario_name, paths in scenarios.items():
        print_distribution_table(scenario_name, paths)