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

def check_position(base_filename):
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
            est = r.get('estimated_position')
            pos = r.get('position')
            root = r.get('root_anchor')

            if est is None or pos is None or root is None:
                skipped += 1
                continue

            # estimated vs true position 유클리드 거리
            err = math.sqrt((est[0] - pos[0])**2 + (est[1] - pos[1])**2)
            total_errors.append(err)
            anchor_errors[root].append(err)

        print(f"\n{'='*50}")
        print(f"[{i}/7] {json_filename}  (유효: {len(total_errors)}개 / 스킵: {skipped}개)")

        if total_errors:
            mae = sum(total_errors) / len(total_errors)
            max_err = max(total_errors)
            min_err = min(total_errors)
            print(f"  [전체]")
            print(f"    평균 오차: {mae:.4f} m")
            print(f"    최소 오차: {min_err:.4f} m")
            print(f"    최대 오차: {max_err:.4f} m")

        print(f"  [root_anchor별]")
        for anchor in sorted(anchor_errors.keys()):
            errs = anchor_errors[anchor]
            mae = sum(errs) / len(errs)
            max_err = max(errs)
            min_err = min(errs)
            ax, ay = ANCHOR_POSITIONS[anchor]
            print(f"    Anchor {anchor} ({len(errs):4d}개) | 평균 {mae:.4f}m | 최소 {min_err:.4f}m | 최대 {max_err:.4f}m | 앵커위치 ({ax}, {ay})")

check_position("log_20260407_231508")