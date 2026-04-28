import random
from itertools import combinations

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from data_loader import load_json_data
from reward import get_reward
from utils import calc_azimuth


class CustomEnv(gym.Env):
    def __init__(self, base_filename):
        super(CustomEnv, self).__init__()

        self.data = load_json_data(base_filename)

        self.action_list = []
        for root in range(1, 7):
            remaining = [j for j in range(1, 7) if j != root]
            for combo in combinations(remaining, 2):
                self.action_list.append([root] + list(combo))
        self.action_space = spaces.Discrete(len(self.action_list))

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(22,), dtype=np.float32
        )

        self.location = 1
        self.current_record = None
        self.prev_azimuth = 0
        self.prev_primary = 1

    def _get_record(self, location):
        records = self.data.get(location, [])
        valid = [r for r in records if r.get('messages') and r.get('estimated_position')]
        if not valid:
            return None
        return random.choice(valid)

    def _build_state(self, record, prev_primary):
        messages = record.get('messages', {})

        overhearing_anchor_ids = [i for i in range(1, 7) if i != prev_primary]

        UINT32 = 1 << 32

        otj = []
        for anchor_id in overhearing_anchor_ids:
            msg = messages.get(str(anchor_id), [0, 0, 0, 0])

            if len(msg) < 4:
                msg = msg + [0] * (4 - len(msg))

            msg = [float(v) for v in msg[:4]]

            # 전부 미수신
            if all(v == 0 for v in msg):
                otj.extend([-1.0, -1.0, -1.0, -1.0])
                continue

            nonzero_ts = [v for v in msg if v != 0]
            base_ts = nonzero_ts[0]

            for v in msg:
                if v == 0:
                    otj.append(-1.0)
                else:
                    rel = (int(v) - int(base_ts)) % UINT32

                    # 0 → 1로 shift
                    otj.append((rel / 1000.0) + 1.0)

        state = [float(self.prev_azimuth), float(prev_primary)] + otj
        return np.array(state, dtype=np.float32)

    def reset(self, seed=None, options=None):
        self.location = random.randint(1, 7)
        self.current_record = self._get_record(self.location)

        if self.current_record is None:
            return self.reset()

        est = self.current_record.get('estimated_position', [0, 0])
        self.prev_azimuth = calc_azimuth(est[0], est[1])
        self.prev_primary = random.randint(1, 6)

        state = self._build_state(self.current_record, self.prev_primary)
        return state, {}

    def step(self, action_idx):
        action = self.action_list[action_idx]
        primary = action[0]

        move = random.choice([-1, 0, 1])
        new_location = max(1, min(7, self.location + move))
        self.location = new_location

        self.current_record = self._get_record(self.location)
        if self.current_record is None:
            return np.zeros(22, dtype=np.float32), 0.0, False, False, {}

        est = self.current_record.get('estimated_position', [0, 0])
        self.prev_azimuth = calc_azimuth(est[0], est[1])
        self.prev_primary = primary

        state = self._build_state(self.current_record, primary)
        reward = get_reward(self.current_record, action)
        return state, reward, False, False, {}

    def get_action_array(self, action_idx):
        return self.action_list[action_idx]
