import os
import json
from collections import defaultdict

def check_anchor_success(base_filename):
    script_dir = os.path.dirname(os.path.abspath(__file__))

    for i in range(1, 8):
        json_path = os.path.join(script_dir, f"{base_filename}_{i}.json")
        if not os.path.exists(json_path):
            print(f"[SKIP] {base_filename}_{i}.json 없음")
            continue

        with open(json_path, 'r', encoding='utf-8') as f:
            records = json.load(f)

        total = len(records)
        anchor_total   = defaultdict(int)  # 메세지 자체가 있는 경우
        anchor_success = defaultdict(int)  # t2,t5,t8 모두 0이 아닌 경우

        for r in records:
            root = r.get('root_anchor')
            for anchor_id_str, msg in r.get('messages', {}).items():
                anchor_id = int(anchor_id_str)
                anchor_total[anchor_id] += 1

                if anchor_id == root:
                    # root는 Ra,Da,Rb,Db,D2b — 모두 0이 아니면 성공
                    if len(msg) >= 5 and all(v != 0 for v in msg[:5]):
                        anchor_success[anchor_id] += 1
                else:
                    # leaf는 t2,t5,t8 모두 0이 아니면 성공
                    if len(msg) >= 3 and msg[0] != 0 and msg[1] != 0 and msg[2] != 0:
                        anchor_success[anchor_id] += 1
        print(f"\n{'='*50}")
        print(f"[{i}/7] {base_filename}_{i}.json  (총 레코드: {total}개)")
        print(f"  {'Anchor':<10} {'수신':>6} {'성공':>6} {'성공률':>8}")
        print(f"  {'-'*34}")
        for anchor in sorted(anchor_total.keys()):
            tot = anchor_total[anchor]
            suc = anchor_success[anchor]
            rate = suc / tot * 100 if tot > 0 else 0
            print(f"  Anchor {anchor:<4}  {tot:>6}  {suc:>6}  {rate:>7.1f}%")

check_anchor_success("log_20260407_231508")