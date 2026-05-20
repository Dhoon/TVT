import random
from collections import namedtuple, deque

import torch
import torch.nn as nn
import torch.nn.functional as F

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, 128)
        self.layer4 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        return self.layer4(x)


class DQN_CNN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN_CNN, self).__init__()

        self.n_scalar = 2
        self.n_channel = 3  # t2, t5, t8
        self.n_leaf = (n_observations - self.n_scalar) // self.n_channel  # 6

        self.cnn = nn.Sequential(
            nn.Conv1d(self.n_channel, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )

        self.scalar_fc = nn.Sequential(
            nn.Linear(self.n_scalar, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
        )

        feature_dim = 128 * self.n_leaf + 64

        self.feature_fc = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )

        # V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        # A(s, a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def forward(self, x):
        scalar = x[:, :self.n_scalar]

        otj = x[:, self.n_scalar:].reshape(-1, self.n_leaf, self.n_channel)
        otj = otj.permute(0, 2, 1)

        cnn_out = self.cnn(otj).flatten(1)
        scalar_out = self.scalar_fc(scalar)

        combined = torch.cat([cnn_out, scalar_out], dim=1)
        feature = self.feature_fc(combined)

        value = self.value_stream(feature)
        advantage = self.advantage_stream(feature)

        q = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q

class DQN_Attention(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN_Attention, self).__init__()
 
        self.n_scalar = 2
        self.n_msg = 3
        self.n_anchor = (n_observations - self.n_scalar) // self.n_msg  # 6
 
        self.embed_dim = 64
        self.anchor_embed = nn.Sequential(
            nn.Linear(self.n_msg, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, self.embed_dim),
        )
 
        self.attention = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=4,
            batch_first=True,
            dropout=0.1,
        )
        self.attn_norm = nn.LayerNorm(self.embed_dim)
 
        self.scalar_fc = nn.Sequential(
            nn.Linear(self.n_scalar, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
        )
 
        feature_dim = self.n_anchor * self.embed_dim + 64
 
        self.feature_fc = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )
 
        self.value_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
 
        self.advantage_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )
 
    def forward(self, x):
        scalar = x[:, :self.n_scalar]
 
        anchors = x[:, self.n_scalar:].reshape(-1, self.n_anchor, self.n_msg)
 
        anchor_emb = self.anchor_embed(anchors)
 
        attn_out, _ = self.attention(anchor_emb, anchor_emb, anchor_emb)
        attn_out = self.attn_norm(attn_out + anchor_emb)
 
        attn_flat = attn_out.flatten(1)
 
        scalar_out = self.scalar_fc(scalar)
 
        combined = torch.cat([attn_flat, scalar_out], dim=1)
        feature = self.feature_fc(combined)
 
        value = self.value_stream(feature)
        advantage = self.advantage_stream(feature)
 
        q = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
