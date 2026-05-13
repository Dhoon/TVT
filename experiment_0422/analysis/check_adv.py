import os
import json
from collections import defaultdict


def get_adv_reception_rate(json_paths):
    """
    여러 json 파일을 합쳐서 각 anchor의 ADV 수신률을 계산합니다.
    수신률 = 해당 anchor가 ADV를 수신한 레코드 수 / 전체 레코드 수
    """
    all_records = []
    for json_path in json_paths:
        if not os.path.exists(json_path):
            print(f"[SKIP] {json_path} 없음")
            continue
        with open(json_path, 'r', encoding='utf-8') as f:
            all_records.extend(json.load(f))

    if not all_records:
        return None

    total = len(all_records)
    adv_count = defaultdict(int)

    for r in all_records:
        adv = r.get('adv', {})
        for anchor_id_str in adv.keys():
            adv_count[int(anchor_id_str)] += 1

    rates = {anchor: count / total * 100 for anchor, count in adv_count.items()}
    return total, rates


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))

    scenarios = {
        "10m (12.64m)": [
            os.path.join(script_dir, "log_20260422_221958_mode1_10.json"),
        ],
        "20m (22.64m)": [
            os.path.join(script_dir, "log_20260422_221958_mode1_20.json"),
        ],
    }

    all_anchors = list(range(1, 7))

    print(f"\n{'='*60}")
    print(f"ADV Reception Rate (%) per Anchor")
    print(f"  {'Scenario':<15} {'Total':>6}" + "".join(f"  A{a:>2}" for a in all_anchors))
    print(f"  {'-'*55}")

    for scenario_name, paths in scenarios.items():
        result = get_adv_reception_rate(paths)
        if result is None:
            continue
        total, rates = result

        row = f"  {scenario_name:<15} {total:>6}"
        for a in all_anchors:
            val = rates.get(a, 0.0)
            row += f"  {val:>4.1f}"
        print(row)