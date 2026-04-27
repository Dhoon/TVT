import os
import json
from collections import defaultdict

def check_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        records = json.load(f)

    total = len(records)
    all_anchors = {'1', '2', '3', '4', '5', '6'}

    # adv 빠진 것
    adv_missing = defaultdict(int)  # anchor -> 빠진 횟수
    # messages 빠진 것
    msg_missing = defaultdict(int)

    for r in records:
        adv_anchors = set(r.get('adv', {}).keys())
        msg_anchors = set(r.get('messages', {}).keys())

        for a in all_anchors:
            if a not in adv_anchors:
                adv_missing[a] += 1
            if a not in msg_anchors:
                msg_missing[a] += 1

    print(f"\n{'='*40}")
    print(f"파일: {os.path.basename(filepath)}")
    print(f"총 레코드: {total}개")

    print(f"\n[ADV 누락]")
    any_adv = False
    for a in sorted(all_anchors):
        cnt = adv_missing[a]
        if cnt > 0:
            print(f"  Anchor {a}: {total}개 중 {cnt}개 누락 ({cnt/total*100:.1f}%)")
            any_adv = True
    if not any_adv:
        print("  누락 없음 ✓")

    print(f"\n[Messages 누락]")
    any_msg = False
    for a in sorted(all_anchors):
        cnt = msg_missing[a]
        if cnt > 0:
            print(f"  Anchor {a}: {total}개 중 {cnt}개 누락 ({cnt/total*100:.1f}%)")
            any_msg = True
    if not any_msg:
        print("  누락 없음 ✓")

def check_all(base_filename):
    script_dir = os.path.dirname(os.path.abspath(__file__))

    for i in range(1, 8):
        json_filename = f"{base_filename}_{i}.json"
        json_path = os.path.join(script_dir, json_filename)

        if not os.path.exists(json_path):
            print(f"[SKIP] {json_filename} 없음")
            continue

        check_json(json_path)

check_all("log_20260407_231508")