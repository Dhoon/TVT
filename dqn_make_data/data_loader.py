import json
import os


def load_json_data(script_dir=None):
    if script_dir is None:
        script_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'test')

    scenarios = {
        1: {
            "label": "Static 12.64m",
            "Round-Robin": (os.path.join(script_dir, "log_20260422_221958_mode1_10.json"), 6),
            "PedLoc":      (os.path.join(script_dir, "log_20260422_224621_mode2_10.json"), None),
        },
        2: {
            "label": "Static 22.64m",
            "Round-Robin": (os.path.join(script_dir, "log_20260422_221958_mode1_20.json"), 6),
            "PedLoc":      (os.path.join(script_dir, "log_20260422_224621_mode2_20.json"), None),
        },
        3: {
            "label": "Static 32.64m",
            "Round-Robin": (os.path.join(script_dir, "log_20260422_221958_mode1_30.json"), 6),
            "PedLoc":      (os.path.join(script_dir, "log_20260422_224621_mode2_30.json"), None),
        },
        4: {
            "label": "Moving (22.64m->2.64m)",
            "Round-Robin": (os.path.join(script_dir, "log_20260422_221958_mode1_20-0.json"), 6),
            "PedLoc":      (os.path.join(script_dir, "log_20260422_224621_mode2_20-0.json"), None),
        },
        5: {
            "label": "Moving (2.64m->22.64m)",
            "Round-Robin": (os.path.join(script_dir, "log_20260422_221958_mode1_0-20.json"), 6),
            "PedLoc":      (os.path.join(script_dir, "log_20260422_224621_mode2_0-20.json"), None),
        },
        6: {
            "label": "NLoS",
            "Round-Robin": (os.path.join(script_dir, "log_20260422_233919_mode1_nlos_1.json"), 6),
            "PedLoc":      (os.path.join(script_dir, "log_20260422_234741_mode2_nlos_1.json"), None),
        },
    }

    data = {}
    pedloc_counts = {}
 
    for i, info in scenarios.items():
        label = info["label"]
        combined = []
        pedloc_count = 0
 
        for mode in ("Round-Robin", "PedLoc"):
            path, num_anchors = info[mode]
            if not os.path.exists(path):
                print(f"[SKIP] location {i} ({label}) [{mode}]: {path} 없음")
                continue
            with open(path, 'r', encoding='utf-8') as f:
                records = json.load(f)
            combined.extend(records)
            print(f"[LOAD] location {i} ({label}) [{mode}]: {len(records)}개 레코드")
            if mode == "PedLoc":
                pedloc_count = len(records)
 
        data[i] = combined
        pedloc_counts[i] = pedloc_count
        print(f"[MERGE] location {i} ({label}): 총 {len(combined)}개 레코드 (PedLoc={pedloc_count})")
 
    return data, pedloc_counts
 
 
if __name__ == "__main__":
    data, pedloc_counts = load_json_data()
    print(f"\n총 로드된 시나리오 수: {len(data)}")
    total = sum(len(v) for v in data.values())
    print(f"총 레코드 수: {total}")
    print(f"PedLoc 개수: {pedloc_counts}")