import os
import json
import glob
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

    # base_filename 뒤에 뭐가 붙든 .json 전부 처리
    pattern = os.path.join(script_dir, f"{base_filename}*.json")
    json_paths = sorted(glob.glob(pattern))

    if not json_paths:
        print(f"[SKIP] {base_filename}*.json 파일 없음")
        return

    for idx, json_path in enumerate(json_paths, start=1):
        json_filename = os.path.basename(json_path)

        with open(json_path, 'r', encoding='utf-8') as f:
            records = json.load(f)

        total_distances = []
        anchor_distances = defaultdict(list)
        skipped = 0

        for r in records:
            root = r.get('root_anchor')
            dist_m = r.get('distance_m')

            if root is None or dist_m is None:
                skipped += 1
                continue

            try:
                root = int(root)
                dist_m = float(dist_m)
            except (TypeError, ValueError):
                skipped += 1
                continue

            if root not in ANCHOR_POSITIONS:
                skipped += 1
                continue

            total_distances.append(dist_m)
            anchor_distances[root].append(dist_m)

        print(f"\n{'='*40}")
        print(
            f"[{idx}/{len(json_paths)}] {json_filename} "
            f"(유효: {len(total_distances)}개 / 스킵: {skipped}개)"
        )

        if total_distances:
            mean_dist = sum(total_distances) / len(total_distances)
            min_dist = min(total_distances)
            max_dist = max(total_distances)

            print(f"  [전체]")
            print(f"    평균 거리: {mean_dist:.4f} m")
            print(f"    최소 거리: {min_dist:.4f} m")
            print(f"    최대 거리: {max_dist:.4f} m")

        print(f"  [앵커별]")
        for anchor in sorted(anchor_distances.keys()):
            dists = anchor_distances[anchor]
            mean_dist = sum(dists) / len(dists)
            min_dist = min(dists)
            max_dist = max(dists)

            print(
                f"    Anchor {anchor} ({len(dists)}개): "
                f"평균 거리 {mean_dist:.4f} m / "
                f"최소 {min_dist:.4f} m / "
                f"최대 {max_dist:.4f} m"
            )

check_distance("log_20260422_214429")
#check_distance("log_20260422_221958_mode1")
#check_distance("log_20260422_233919_mode2")