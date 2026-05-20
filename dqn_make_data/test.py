import copy
import json
import os
from datetime import datetime

import torch

from env import CustomEnv, record_fail_count
from model import DQN_CNN,DQN_Attention
from reward import GROUND_TRUTH

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

script_dir = os.path.dirname(os.path.abspath(__file__))

env = CustomEnv()

n_actions = env.action_space.n
state, info = env.reset()
n_observations = len(state)

# policy_net = DQN_CNN(n_observations, n_actions).to(device)
# policy_net.load_state_dict(torch.load('policy_net_CNN.pth', map_location=device))
policy_net = DQN_Attention(n_observations, n_actions).to(device)
policy_net.load_state_dict(torch.load('policy_net_ft.pth', map_location=device))
policy_net.eval()


N_SCALAR = 2
UNAVAIL_PENALTY = 10.0

def select_action(state):
    with torch.no_grad():
        q_values = policy_net(state)
        state_np = state.cpu().numpy()[0]
        for anchor_idx in range(6):
            offset = N_SCALAR + anchor_idx * 3
            if all(state_np[offset + k] <= -0.9 for k in range(3)):
                for i, act in enumerate(env.action_list):
                    if act[0] == anchor_idx + 1:
                        q_values[0][i] -= UNAVAIL_PENALTY
        return q_values.max(1).indices.view(1, 1)


SCENARIO_LABELS = {
    1: "Static 12.64m",
    2: "Static 22.64m",
    3: "Static 32.64m",
    4: "Moving (22.64m->2.64m)",
    5: "Moving (2.64m->22.64m)",
    6: "NLoS",
}

if __name__ == "__main__":
    os.makedirs('log', exist_ok=True)
    os.makedirs('result', exist_ok=True)
    log_path = f"log/test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    log_file = open(log_path, 'w', encoding='utf-8')

    def log(msg):
        print(msg)
        log_file.write(msg + '\n')
        log_file.flush()

    log(f"{'='*60}")
    log(f"Test Start | device={device}")
    log(f"{'='*60}")

    for location in range(1, 7):
        label = SCENARIO_LABELS[location]
        max_step = env.pedloc_counts[location]
        true_pos = GROUND_TRUTH.get(location)

        env.location = location
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

        ep_cases = {}
        ep_sums = {'Rprimary': 0.0, 'Rangle': 0.0, 'Rbest': 0.0, 'rdop': 0.0}
        ep_n_success = 0
        ep_n_all = 0
        total_reward = 0.0
        scenario_records = []
        seq = 0

        # primary 통계
        primary_select_count = {i: 0 for i in range(1, 7)}   # 선택 횟수
        primary_success_count = {i: 0 for i in range(1, 7)}  # 성공 횟수 (record 있음 + distance_m 있음)

        log(f"\n[Location {location}] {label} | steps={max_step}")
        log(f"{'-'*50}")

        for t in range(max_step):
            action = select_action(state)
            action_arr = env.get_action_array(action.item())
            primary = action_arr[0]

            observation, reward, terminated, truncated, info = env.step(action.item())
            total_reward += reward

            state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            # primary 통계 집계
            primary_select_count[primary] += 1
            if not info.get('fail'):
                record = info.get('record', {})
                if record and record.get('distance_m') is not None:
                    primary_success_count[primary] += 1

            # JSON 레코드 구성
            leaf = action_arr[1:]
            if info.get('fail'):
                json_record = {
                    "seq": seq % 256,
                    "tag": 257,
                    "timestamp": None,
                    "root_anchor": primary,
                    "leaf": leaf,
                    "distance_m": None,
                    "estimated_position": None,
                    "position": list(true_pos),
                    "adv": None,
                    "messages": None,
                }
            else:
                record = info['record']
                json_record = copy.deepcopy(record)
                json_record['seq'] = seq % 256
                json_record['leaf'] = leaf
                json_record['position'] = list(true_pos) if true_pos is not None else None
                json_record['estimated_position'] = info.get('estimated_position')

            scenario_records.append(json_record)
            seq += 1

            if 'case' in info:
                case = info['case']
                ep_cases[case] = ep_cases.get(case, 0) + 1
                ep_sums['Rprimary'] += info['Rprimary']
                ep_n_all += 1
                if case == 'success':
                    for k in ('Rangle', 'Rbest', 'rdop'):
                        ep_sums[k] += info[k]
                    ep_n_success += 1

            if terminated:
                break

        # 시나리오 JSON 저장
        result_path = f"result/location_{location}_{label.replace(' ', '_').replace('>', 'to')}.json"
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(scenario_records, f, ensure_ascii=False, indent=2)
        log(f"  [SAVED] {result_path} ({len(scenario_records)}개 레코드)")

        na = ep_n_all if ep_n_all > 0 else 1
        ns = ep_n_success if ep_n_success > 0 else 1
        case_str = '  '.join(f"{k}={v}" for k, v in sorted(ep_cases.items()))
        avg_str = (f"Rprimary={ep_sums['Rprimary']/na:+.3f}(all)"
                   f"  Rangle={ep_sums['Rangle']/ns:+.3f}"
                   f"  Rbest={ep_sums['Rbest']/ns:+.3f}"
                   f"  rdop={ep_sums['rdop']/ns:+.3f}")
        tot_str = '  '.join(f"{k}={ep_sums[k]:+.1f}" for k in ('Rprimary', 'Rangle', 'Rbest', 'rdop'))

        log(f"  reward={total_reward:+7.2f} | success={ep_n_success}/{ep_n_all}")
        log(f"  cases : {case_str}")
        log(f"  avg   : {avg_str}")
        log(f"  total : {tot_str}")

        # Primary Anchor Selection Distribution
        total_selected = sum(primary_select_count.values())
        log(f"\n  [Primary Anchor Selection Distribution]")
        for i in range(1, 7):
            count = primary_select_count[i]
            ratio = count / total_selected * 100 if total_selected > 0 else 0
            log(f"    Anchor {i}: {count:4d} ({ratio:5.1f}%)")

        # Primary Anchor Communication Success Rate
        log(f"\n  [Primary Anchor Communication Success Rate]")
        for i in range(1, 7):
            selected = primary_select_count[i]
            success = primary_success_count[i]
            rate = success / selected * 100 if selected > 0 else 0
            log(f"    Anchor {i}: {success:4d}/{selected:4d} ({rate:5.1f}%)")
        
        import env as env_module
        log(f"\n총 record_fail_count: {env_module.record_fail_count}")

    log(f"\n{'='*60}")
    log("Test Complete")
    log(f"{'='*60}")
    log_file.close()
    env.close()