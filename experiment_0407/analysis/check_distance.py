import os
import json
import math
from collections import defaultdict

ANCHOR_POSITIONS = {
    1: (-0.75, 1.50),
    2: ( 0.75, 1.50),
    3: (-0.75, 0.00),
    4: ( 0.75, 0.00),
    5: (-0.57, -0.30),
    6: ( 0.57, -0.30),
}

def check_distance(base_filename):
    script_dir = os.path.dirname(os.path.abspath(__file__))

    for i in range(1, 8):
        json_filename = f"{base_filename}_{i}.json"
        json_path = os.path.join(script_dir, json_filename)

        if not os.path.exists(json_path):
            print(f"[SKIP] {json_filename} 없음")
            continue

        with open(json_path, 'r', encoding='utf-8') as f:
            records = json.load(f)

        total_errors = []
        anchor_errors = defaultdict(list)
        skipped = 0

        for r in records:
            root = r.get('root_anchor')
            dist_m = r.get('distance_m')
            position = r.get('position')

            if root is None or dist_m is None or position is None:
                skipped += 1
                continue

            if root not in ANCHOR_POSITIONS:
                skipped += 1
                continue

            ax, ay = ANCHOR_POSITIONS[root]
            px, py = position
            true_dist = math.sqrt((px - ax)**2 + (py - ay)**2)
            error = dist_m - true_dist

            total_errors.append(error)
            anchor_errors[root].append(error)

        print(f"\n{'='*40}")
        print(f"[{i}/7] {json_filename}  (유효: {len(total_errors)}개 / 스킵: {skipped}개)")

        if total_errors:
            mae = sum(abs(e) for e in total_errors) / len(total_errors)
            mean_err = sum(total_errors) / len(total_errors)
            max_err = max(total_errors, key=abs)
            print(f"  [전체]")
            print(f"    평균 오차:      {mean_err:+.4f} m")
            print(f"    평균 절대 오차: {mae:.4f} m")
            print(f"    최대 오차:      {max_err:+.4f} m")

        print(f"  [앵커별]")
        for anchor in sorted(anchor_errors.keys()):
            errs = anchor_errors[anchor]
            mae = sum(abs(e) for e in errs) / len(errs)
            mean_err = sum(errs) / len(errs)
            max_err = max(errs, key=abs)
            print(f"    Anchor {anchor} ({len(errs)}개): 평균 오차 {mean_err:+.4f} m / MAE {mae:.4f} m / 최대 {max_err:+.4f} m")

check_distance("log_20260407_231508")