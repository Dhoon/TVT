# Ablation study: w/o R_geom
# reward2.get_reward is patched into env before training starts
import math
import random
import os
from datetime import datetime

import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

import env as _env_mod
import reward2
_env_mod.get_reward = reward2.get_reward

from env import CustomEnv
from model import DQN, DQN_CNN, DQN_Attention, ReplayMemory, Transition
from settings import (BATCH_SIZE, EPS_DECAY, EPS_END, EPS_START,
                      EPISODES, GAMMA, LR, MAX_STEP, TAU,
                      FINETUNE_EPISODES, FINETUNE_DROP_PROB,
                      FINETUNE_EPS_START, FINETUNE_LR)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = CustomEnv("log_20260407_231508")

n_actions = env.action_space.n
state, info = env.reset()
n_observations = len(state)

policy_net = DQN_Attention(n_observations, n_actions).to(device)
target_net = DQN_Attention(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(3000)
steps_done = 0
_eps_start = EPS_START


N_SCALAR = 2

UNAVAIL_PENALTY = 10.0

def _mask_invalid_actions(q_values, state):
    state_np = state.cpu().numpy()[0]
    for anchor_idx in range(6):
        offset = N_SCALAR + anchor_idx * 3
        if all(state_np[offset + k] <= -0.9 for k in range(3)):
            for i, act in enumerate(env.action_list):
                if act[0] == anchor_idx + 1:
                    q_values[0][i] -= UNAVAIL_PENALTY
    return q_values


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (_eps_start - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            q_values = policy_net(state)
            q_values = _mask_invalid_actions(q_values, state)
            return q_values.max(1).indices.view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(
        tuple(map(lambda s: s is not None, batch.next_state)),
        device=device, dtype=torch.bool
    )
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_actions = policy_net(non_final_next_states).argmax(1, keepdim=True)

        next_state_values[non_final_mask] = target_net(non_final_next_states) \
            .gather(1, next_actions) \
            .squeeze(1)
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display
plt.ion()

episode_rewards = []


def plot_progress():
    plt.figure(1)
    plt.clf()
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('DQN Training Progress (w/o R_geom)')
    plt.plot(episode_rewards, label='Episode Reward')
    if len(episode_rewards) >= 10:
        rewards_t = torch.tensor(episode_rewards, dtype=torch.float)
        moving_avg = rewards_t.unfold(0, 10, 1).mean(1).view(-1)
        moving_avg = torch.cat((torch.zeros(9), moving_avg))
        plt.plot(moving_avg.numpy(), label='Moving Avg (10)')
    plt.legend()
    plt.pause(0.001)
    if is_ipython:
        display.display(plt.gcf())
        display.clear_output(wait=True)


def run_episodes(n_episodes, stage_label, total_offset=0):
    global steps_done, _eps_start
    for i_episode in range(n_episodes):
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        total_reward = 0.0

        ep_cases = {}
        ep_sums = {'Rprimary': 0.0, 'Rangle': 0.0, 'Rbest': 0.0, 'R_geom': 0.0}

        ep_n_success = 0
        ep_n_all = 0

        for _ in range(MAX_STEP):
            az = state[0][0].item()
            prev_p = int(state[0][1].item())
            action = select_action(state)
            action_arr = env.get_action_array(action.item())
            observation, reward, _, _, info = env.step(action.item())
            log(f"[STEP] az={az:.2f} prev_primary={prev_p} action={action_arr} reward={reward:+.3f}")
            total_reward += reward
            reward = torch.tensor([reward], device=device)
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            if 'case' in info:
                case = info['case']
                ep_cases[case] = ep_cases.get(case, 0) + 1
                ep_sums['Rprimary'] += info['Rprimary']
                ep_sums['R_geom']   += info['R_geom']
                ep_n_all += 1
                if case == 'success':
                    for k in ('Rangle', 'Rbest'):
                        ep_sums[k] += info[k]
                    ep_n_success += 1

            memory.push(state, action, next_state, reward)
            state = next_state

            optimize_model()

            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (1 - TAU)
            target_net.load_state_dict(target_net_state_dict)

        episode_rewards.append(total_reward)

        na = ep_n_all if ep_n_all > 0 else 1
        ns = ep_n_success if ep_n_success > 0 else 1
        case_str = '  '.join(f"{k}={v}" for k, v in sorted(ep_cases.items()))
        avg_str  = (f"Rprimary={ep_sums['Rprimary']/na:+.3f}(all)"
                    f"  Rangle={ep_sums['Rangle']/ns:+.3f}"
                    f"  Rbest={ep_sums['Rbest']/ns:+.3f}"
                    f"  R_geom(ref)={ep_sums['R_geom']/na:+.3f}")
        tot_str  = '  '.join(f"{k}={ep_sums[k]:+.1f}" for k in ('Rprimary', 'Rangle', 'Rbest', 'R_geom'))
        ep_num = total_offset + i_episode + 1
        ep_total = total_offset + n_episodes
        log(f"[{stage_label}] Ep {ep_num:4d}/{ep_total} | reward={total_reward:+7.2f} | success={ep_n_success}/{ep_n_all}")
        log(f"  cases : {case_str}")
        log(f"  avg   : {avg_str}")
        log(f"  total : {tot_str}")
        plot_progress()


if __name__ == "__main__":
    os.makedirs('log', exist_ok=True)
    log_path = f"log/train_wo_rgeom_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    log_file = open(log_path, 'w', encoding='utf-8')

    def log(msg):
        print(msg)
        log_file.write(msg + '\n')
        log_file.flush()

    # Stage 1
    if os.path.exists('policy_net_wo_rgeom.pth'):
        policy_net.load_state_dict(torch.load('policy_net_wo_rgeom.pth', map_location=device))
        target_net.load_state_dict(policy_net.state_dict())
        log('Stage 1 skipped. Loaded policy_net_wo_rgeom.pth')
    else:
        log(f"{'='*60}")
        log(f"Stage 1 | episodes={EPISODES} drop={env.anchor_drop_prob} lr={LR}")
        log(f"{'='*60}")
        run_episodes(EPISODES, 'S1', total_offset=0)
        torch.save(policy_net.state_dict(), 'policy_net_wo_rgeom.pth')
        log('Stage 1 complete. Saved policy_net_wo_rgeom.pth')

    # Stage 2
    steps_done = 0
    _eps_start = FINETUNE_EPS_START
    env.anchor_drop_prob = FINETUNE_DROP_PROB
    memory.__init__(3000)
    for g in optimizer.param_groups:
        g['lr'] = FINETUNE_LR

    log(f"{'='*60}")
    log(f"Stage 2 | episodes={FINETUNE_EPISODES} drop={FINETUNE_DROP_PROB} lr={FINETUNE_LR}")
    log(f"{'='*60}")
    run_episodes(FINETUNE_EPISODES, 'S2', total_offset=EPISODES)
    torch.save(policy_net.state_dict(), 'policy_net_ft_wo_rgeom.pth')
    log('Stage 2 complete. Saved policy_net_ft_wo_rgeom.pth')

    log_file.close()
    plt.ioff()
    plt.show()
    env.close()
