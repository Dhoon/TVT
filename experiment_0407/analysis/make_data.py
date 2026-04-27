import re
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
    1: (4.75, 0.00),
    2: (3.17, 2.55),
    3: (1.58, 5.09),
    4: (0.00, 7.64),
    5: (-1.58, 5.09),
    6: (-3.17, 2.55),
    7: (-4.75, 0.00),
}

C = 299702547
DWT_TIME_UNIT = 1 / (499.2e6 * 128.0)

# ── 1. 파싱 ──────────────────────────────────────────────
def parse_timestamp_str(line):
    match = re.match(r'(\d{2}:\d{2}:\d{2}:\d{3}):\d{3}', line)
    return match.group(1) if match else None

def parse_log_to_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    records = []
    current = None

    for line in lines:
        line = line.strip()

        adv_match = re.search(r'\[ADV\] From Anchor (\d+), Tag (\d+), Seq=(\d+), Power=(\d+), CIR=\[([^\]]+)\]', line)
        if adv_match:
            anchor, tag, seq, power, cir_str = adv_match.groups()
            anchor, tag, seq, power = int(anchor), int(tag), int(seq), int(power)
            cir = list(map(int, cir_str.split(', ')))

            if current is None or current['seq'] != seq:
                if current is not None:
                    records.append(current)
                current = {
                    "seq": seq,
                    "tag": tag,
                    "timestamp": parse_timestamp_str(line),
                    "root_anchor": None,
                    "distance_m": None,
                    "estimated_position": None,
                    "position": None,
                    "adv": {},
                    "messages": {}
                }

            current['adv'][str(anchor)] = {"power": power, "cir": cir}
            continue

        root_match = re.search(r'\[ROOT SELECTED\] Anchor (\d+) for Tag \d+ Seq (\d+)', line)
        if root_match and current is not None:
            current['root_anchor'] = int(root_match.group(1))
            continue

        leaf_match = re.search(r'\[Leaf\] Type \d+, Anchor (\d+), Tag \d+, Seq=(\d+), message = \[([^\]]+)\]', line)
        if leaf_match and current is not None:
            anchor = str(int(leaf_match.group(1)))
            msg = list(map(int, leaf_match.group(3).split(', ')))
            current['messages'][anchor] = msg
            continue

        dstwr_match = re.search(r'\[DS-TWR\] Type \d+, Anchor (\d+), Tag \d+, Seq=(\d+).*message = \[([^\]]+)\]', line)
        if dstwr_match and current is not None:
            anchor = str(int(dstwr_match.group(1)))
            msg = list(map(int, dstwr_match.group(3).split(', ')))
            current['messages'][anchor] = msg
            continue

        tof_match = re.search(r'\[TOF\] Root TOF:.*Distance: ([\d.]+) m', line)
        if tof_match and current is not None:
            current['distance_m'] = float(tof_match.group(1))
            continue

        # estimated_position은 여기서 읽지 않고 아래에서 재계산

    if current is not None:
        records.append(current)

    return records

# ── 2. estimated_position 재계산 ─────────────────────────
def estimate_position(record):
    root_anchor_id = record.get('root_anchor')
    messages = record.get('messages', {})

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

    leaf_positions = []
    tdoa_deltas = []

    for anchor_id_str, msg in messages.items():
        anchor_id = int(anchor_id_str)
        if anchor_id == root_anchor_id:
            continue

        t2, t5, t8, t11 = msg[:4]

        if t2 == 0 or t5 == 0 or t8 == 0:
            continue

        UINT32 = 1 << 32
        Rt1 = (t5 - t2) % UINT32
        Rt2 = (t8 - t5) % UINT32

        denominator = 2 * Rt1 + 2 * Rt2
        numerator = Da * Rt1 - Rt2 * Ra + Rb * Rt1 - Rt2 * Db
        tdoa_dtu = numerator / denominator if denominator != 0 else 0
        tdoa = tdoa_dtu * DWT_TIME_UNIT * C

        pos = ANCHOR_POSITIONS.get(anchor_id)
        if pos:
            leaf_positions.append(pos)
            tdoa_deltas.append(tdoa)

    def residuals(p):
        x, y = p
        res = [np.linalg.norm([x - root_pos[0], y - root_pos[1]]) - root_dist]
        for (lx, ly), delta_d in zip(leaf_positions, tdoa_deltas):
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

    est_x, est_y = result.x
    return [round(est_x, 4), round(est_y, 4)]

# ── 3. 전체 파이프라인 ────────────────────────────────────
def process_all(base_filename):
    script_dir = os.path.dirname(os.path.abspath(__file__))

    for i in range(1, 8):
        txt_filename = f"{base_filename}_{i}.txt"
        txt_path = os.path.join(script_dir, txt_filename)

        if not os.path.exists(txt_path):
            print(f"[SKIP] {txt_filename} 없음")
            continue

        # 1. 파싱
        records = parse_log_to_json(txt_path)

        # 2. position 채우기
        pos = list(TAG_POSITIONS[i])
        for r in records:
            r['position'] = pos

        # 3. estimated_position 재계산
        success = 0
        failed = 0
        for r in records:
            est = estimate_position(r)
            r['estimated_position'] = est
            if est is not None:
                success += 1
            else:
                failed += 1

        # 4. JSON 저장
        json_filename = f"{base_filename}_{i}.json"
        json_path = os.path.join(script_dir, json_filename)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(records, f, indent=2, ensure_ascii=False)

        print(f"[{i}/7] {txt_filename} → {json_filename} | 레코드 {len(records)}개 | 추정 성공 {success}개 / 실패 {failed}개")

process_all("log_20260407_231508")