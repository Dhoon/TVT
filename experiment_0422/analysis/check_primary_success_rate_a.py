import os
import json


def check_primary_success(json_path):
    """
    주어진 json 파일에서 root_anchor(primary)의 통신 성공률을 계산합니다.
    성공 조건: messages[root_anchor] 5개 값이 모두 0이 아닌 경우
    """
    if not os.path.exists(json_path):
        print(f"[SKIP] {json_path} 없음")
        return None

    with open(json_path, 'r', encoding='utf-8') as f:
        records = json.load(f)

    total   = 0
    success = 0

    for r in records:
        root = r.get('root_anchor')
        if root is None:
            continue

        messages = r.get('messages', {})
        root_msg = messages.get(str(root))

        total += 1

        # 5개 값이 존재하고 모두 0이 아니면 성공
        if root_msg and len(root_msg) >= 5 and all(v != 0 for v in root_msg[:5]):
            success += 1

    rate = success / total * 100 if total > 0 else 0
    return total, success, rate


def compare_methods(scenario_name, paths: dict):
    print(f"\n{'='*55}")
    print(f"Scenario: {scenario_name}")
    print(f"  {'Method':<15} {'Total':>8} {'Success':>8} {'Rate':>8}")
    print(f"  {'-'*41}")

    results = {}
    for method, path in paths.items():
        result = check_primary_success(path)
        if result is None:
            print(f"  {method:<15} {'N/A':>8}")
            continue
        total, success, rate = result
        print(f"  {method:<15} {total:>8} {success:>8} {rate:>7.1f}%")
        results[method] = rate

    return results


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # -------------------------------------------------------
    # mode1 = Round-Robin
    # mode2 = PedLoc
    # 거리 표기: 차량 중앙 기준 + 2.64m 오프셋 적용
    #   10m -> 12.64m, 20m -> 22.64m, 30m -> 32.64m
    #   Moving 20->0: 22.64m->2.64m
    #   Moving 0->20: 2.64m->22.64m
    # DA-VPL 추후 추가
    # -------------------------------------------------------
    scenarios = {
        "Static 12.64m": {
            "Round-Robin": os.path.join(script_dir, "log_20260422_221958_mode1_10.json"),
            "PedLoc":      os.path.join(script_dir, "log_20260422_224621_mode2_10.json"),
        },
        "Static 22.64m": {
            "Round-Robin": os.path.join(script_dir, "log_20260422_221958_mode1_20.json"),
            "PedLoc":      os.path.join(script_dir, "log_20260422_224621_mode2_20.json"),
        },
        "Static 32.64m": {
            "Round-Robin": os.path.join(script_dir, "log_20260422_221958_mode1_30.json"),
            "PedLoc":      os.path.join(script_dir, "log_20260422_224621_mode2_30.json"),
        },
        "Moving (22.64m->2.64m)": {
            "Round-Robin": os.path.join(script_dir, "log_20260422_221958_mode1_20-0.json"),
            "PedLoc":      os.path.join(script_dir, "log_20260422_224621_mode2_20-0.json"),
        },
        "Moving (2.64m->22.64m)": {
            "Round-Robin": os.path.join(script_dir, "log_20260422_221958_mode1_0-20.json"),
            "PedLoc":      os.path.join(script_dir, "log_20260422_224621_mode2_0-20.json"),
        },
        "NLoS": {
            "Round-Robin": os.path.join(script_dir, "log_20260422_233919_mode1_nlos_1.json"),
            "PedLoc":      os.path.join(script_dir, "log_20260422_234741_mode2_nlos_1.json"),
        },
    }

    all_results = {}
    for scenario_name, paths in scenarios.items():
        all_results[scenario_name] = compare_methods(scenario_name, paths)

    # 전체 평균
    print(f"\n{'='*55}")
    print("Average across all scenarios:")
    methods = ["Round-Robin", "PedLoc"]
    print(f"  {'Method':<15} {'Avg Rate':>8}")
    print(f"  {'-'*25}")
    for method in methods:
        rates = [all_results[s][method] for s in all_results if method in all_results[s]]
        avg = sum(rates) / len(rates) if rates else 0
        print(f"  {method:<15} {avg:>7.1f}%")