import math
import random

import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

from env import CustomEnv
from model import DQN, DQN_CNN, ReplayMemory, Transition
from settings import (BATCH_SIZE, EPS_DECAY, EPS_END, EPS_START,
                      EPISODES, GAMMA, LR, MAX_STEP, TAU)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = CustomEnv("log_20260407_231508")

n_actions = env.action_space.n
state, info = env.reset()
n_observations = len(state)

policy_net = DQN_CNN(n_observations,n_actions).to(device)
target_net = DQN_CNN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(3000)
steps_done = 0


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1).indices.view(1, 1)
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
    plt.title('DQN Training Progress')
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


if __name__ == "__main__":
    for i_episode in range(EPISODES):
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        total_reward = 0.0

        for t in range(MAX_STEP):
            action = select_action(state)
            observation, reward, terminated, truncated, _ = env.step(action.item())
            total_reward += reward
            reward = torch.tensor([reward], device=device)
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            memory.push(state, action, next_state, reward)
            state = next_state

            optimize_model()

            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (1 - TAU)
            target_net.load_state_dict(target_net_state_dict)

        episode_rewards.append(total_reward)
        print(f"Episode {i_episode+1}/{EPISODES} - Total Reward: {total_reward:.2f}")
        plot_progress()

    print('Complete')
    plt.ioff()
    plt.show()
    env.close()
