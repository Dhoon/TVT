import os
import json
import numpy as np
from scipy.optimize import least_squares
from collections import defaultdict

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
}

C = 299702547
DWT_TIME_UNIT = 1 / (499.2e6 * 128.0)


def compute_gdop(tag_pos, anchor_ids):
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
        return np.sqrt(np.trace(HtH_inv))
    except np.linalg.LinAlgError:
        return None


def get_best_gdop_leaf(tag_pos, root):
    candidates = [a for a in ANCHOR_POSITIONS.keys() if a != root]
    best_gdop = float('inf')
    best_leaf = None
    for leaf in candidates:
        gdop = compute_gdop(tag_pos, [root, leaf])
        if gdop is not None and gdop < best_gdop:
            best_gdop = gdop
            best_leaf = leaf
    return best_leaf, best_gdop


def estimate_position(root_anchor_id, root_msg, leaf_id, messages, init_pos):
    Ra, Da, Rb, Db, D2b = root_msg[:5]
    if any(ts < 0 for ts in (Ra, Da, Rb, Db, D2b)):
        return None

    numerator   = Ra * Rb - Da * Db
    denominator = Ra + Rb + Da + Db
    tof_dtu     = numerator / denominator if denominator != 0 else 0
    root_dist   = tof_dtu * DWT_TIME_UNIT * C
    root_pos    = ANCHOR_POSITIONS[root_anchor_id]

    msg = messages.get(str(leaf_id))
    if not msg or len(msg) < 3:
        return None

    t2, t5, t8 = msg[0], msg[1], msg[2]
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
        r1 = np.linalg.norm([x - lx, y - ly]) - np.linalg.norm([lx - root_pos[0], ly - root_pos[1]]) - tdoa
        return [r0, r1]

    result = least_squares(residuals, x0=init_pos)
    if not result.success:
        return None
    return result.x


def analyze_power_threshold(json_path, tag_pos, power_bin_size=5):
    if not os.path.exists(json_path):
        print(f"[SKIP] {json_path} 없음")
        return

    with open(json_path, 'r', encoding='utf-8') as f:
        records = json.load(f)

    data = []
    prev_pos = list(tag_pos)

    skip_no_adv_power  = 0
    skip_no_est        = 0
    best_leaf_counter  = defaultdict(int)
    power_none_counter = defaultdict(int)

    for r in records:
        root = r.get('root_anchor')
        if root is None:
            continue

        messages = r.get('messages') or {}
        root_msg = messages.get(str(root))

        if not (root_msg and len(root_msg) >= 5 and all(v != 0 for v in root_msg[:5])):
            continue

        adv = r.get('adv') or {}

        best_leaf, best_gdop = get_best_gdop_leaf(tag_pos, root)
        if best_leaf is None:
            continue

        best_leaf_counter[best_leaf] += 1

        leaf_power = adv.get(str(best_leaf), {}).get('power', None)
        if leaf_power is None:
            power_none_counter[best_leaf] += 1
            skip_no_adv_power += 1
            continue

        est = estimate_position(root, root_msg, best_leaf, messages, prev_pos)
        if est is None:
            skip_no_est += 1
            continue

        prev_pos = list(est)
        error = np.linalg.norm([est[0] - tag_pos[0], est[1] - tag_pos[1]])

        data.append({
            "power": leaf_power,
            "error": error,
            "gdop":  best_gdop,
            "root":  root,
            "leaf":  best_leaf,
        })

    # 디버그 정보
    print(f"\n  [DEBUG] best leaf 선택 분포: {dict(sorted(best_leaf_counter.items()))}")
    print(f"  [DEBUG] power=None 스킵 (leaf별): {dict(sorted(power_none_counter.items()))}")
    print(f"  [DEBUG] skip_no_adv_power={skip_no_adv_power}, skip_no_est={skip_no_est}")
    print(f"  [DEBUG] 최종 분석 레코드 수: {len(data)}")

    if not data:
        print("  결과 없음")
        return

    powers = np.array([d['power'] for d in data])
    errors = np.array([d['error'] for d in data])

    # 파워 전체 분포
    print(f"\n  파워 전체 분포: min={powers.min()}, max={powers.max()}, mean={powers.mean():.1f}")
    unique, counts = np.unique(powers, return_counts=True)
    print(f"  파워값별 count: { {int(u): int(c) for u, c in zip(unique, counts)} }")

    # 파워 구간별 통계
    min_p = int(powers.min())
    max_p = int(powers.max())
    bins  = range(min_p, max_p + power_bin_size, power_bin_size)

    print(f"\n  파워 구간별 위치측위 오차 (bin size={power_bin_size})")
    print(f"  {'Power range':>15} {'Count':>6} {'Mean err':>10} {'Std err':>10} {'Median err':>12}")
    print(f"  {'-'*58}")

    for b in bins:
        mask = (powers >= b) & (powers < b + power_bin_size)
        if mask.sum() == 0:
            continue
        e = errors[mask]
        print(
            f"  {b:>6}~{b+power_bin_size-1:<6}   "
            f"{mask.sum():>6} "
            f"{e.mean():>10.4f} "
            f"{e.std():>10.4f} "
            f"{np.median(e):>12.4f}"
        )

    print(f"\n  전체: count={len(data)}, mean={errors.mean():.4f}, std={errors.std():.4f}")

    # threshold 분석
    print(f"\n  파워 threshold별 오차 비교 (threshold 이상 vs 미만)")
    print(f"  {'Threshold':>10} {'Above cnt':>10} {'Above mean':>12} {'Below cnt':>10} {'Below mean':>12}")
    print(f"  {'-'*58}")
    thresholds = range(min_p + power_bin_size, max_p, power_bin_size)
    for t in thresholds:
        above = errors[powers >= t]
        below = errors[powers <  t]
        if len(above) == 0 or len(below) == 0:
            continue
        print(
            f"  {t:>10} "
            f"{len(above):>10} {above.mean():>12.4f} "
            f"{len(below):>10} {below.mean():>12.4f}"
        )


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))

    scenarios = {
        "Static 12.64m": os.path.join(script_dir, "log_20260422_224621_mode2_10.json"),
        "Static 22.64m": os.path.join(script_dir, "log_20260422_224621_mode2_20.json"),
        "Static 32.64m": os.path.join(script_dir, "log_20260422_224621_mode2_30.json"),
    }

    for scenario_name, path in scenarios.items():
        tag_pos = TAG_POSITIONS[scenario_name]
        print(f"\n{'='*60}")
        print(f"Scenario: {scenario_name}  |  Tag GT: {tag_pos}")
        analyze_power_threshold(path, tag_pos, power_bin_size=5)