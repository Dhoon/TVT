import os
import json

POSITIONS = {
    1: (4.75, 0.00),
    2: (3.17, 2.55),
    3: (1.58, 5.09),
    4: (0.00, 7.64),
    5: (-1.58, 5.09),
    6: (-3.17, 2.55),
    7: (-4.75, 0.00),
}

def fill_position(base_filename):
    script_dir = os.path.dirname(os.path.abspath(__file__))

    for i in range(1, 8):
        json_filename = f"{base_filename}_{i}.json"
        json_path = os.path.join(script_dir, json_filename)

        if not os.path.exists(json_path):
            print(f"[SKIP] {json_filename} 없음")
            continue

        with open(json_path, 'r', encoding='utf-8') as f:
            records = json.load(f)

        pos = list(POSITIONS[i])
        for r in records:
            r['position'] = pos

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(records, f, indent=2, ensure_ascii=False)

        print(f"[{i}/7] {json_filename} → position {pos} 적용 ({len(records)}개 레코드)")

fill_position("log_20260407_231508")