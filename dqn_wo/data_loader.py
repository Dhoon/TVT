import json
import os


def load_json_data(base_filename, num_files=7):
    data = {}
    training_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'training')
    for i in range(1, num_files + 1):
        path = os.path.join(training_dir, f"{base_filename}_{i}.json")
        if not os.path.exists(path):
            print(f"[SKIP] {path} 없음")
            continue
        with open(path, 'r', encoding='utf-8') as f:
            records = json.load(f)
        data[i] = records
        print(f"[LOAD] location {i}: {len(records)}개 레코드")
    return data
