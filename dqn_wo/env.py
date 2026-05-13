import random

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from data_loader import load_json_data
from reward import get_reward


class CustomEnv(gym.Env):
    def __init__(self, base_filename):
        super(CustomEnv, self).__init__()

        self.data = load_json_data(base_filename)

        # action: primary anchor only
        self.action_list = [1, 2, 3, 4, 5, 6]
        self.action_space = spaces.Discrete(len(self.action_list))

        # state: prev_azimuth + prev_primary + OTJ(5 anchors x 4 timestamps)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(22,), dtype=np.float32
        )

        self.location = 1
        self.current_record = None
        self.prev_azimuth = 0.0
        self.prev_primary = 1

    def _get_record(self, location, primary=None):
        records = self.data.get(location, [])
        valid = [r for r in records if r.get('messages')]

        if primary is not None:
            valid = [r for r in valid if r.get('root_anchor') == primary]

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

            if all(v == 0 for v in msg):
                otj.extend([-1.0, -1.0, -1.0, -1.0])
                continue

            base_ts = next(v for v in msg if v != 0)

            for v in msg:
                if v == 0:
                    otj.append(-1.0)
                else:
                    rel = (v - base_ts) % UINT32
                    otj.append((rel / 1000.0) + 1.0)

        state = [float(self.prev_azimuth), float(prev_primary)] + otj
        return np.array(state, dtype=np.float32)

    def reset(self, seed=None, options=None):
        self.location = random.randint(1, 7)
        self.prev_primary = random.randint(1, 6)

        self.current_record = self._get_record(self.location, self.prev_primary)

        if self.current_record is None:
            return self.reset(seed=seed, options=options)

        # primary-only ablation에서는 leaf 기반 위치추정 제거
        self.prev_azimuth = 0.0

        state = self._build_state(self.current_record, self.prev_primary)
        return state, {}

    def step(self, action_idx):
        primary = self.action_list[action_idx]

        move = random.choice([-1, 0, 1])
        self.location = max(1, min(7, self.location + move))

        self.current_record = self._get_record(self.location, primary)

        if self.current_record is None:
            state = np.zeros(22, dtype=np.float32)
            info = {
                'case': 'primary_fail',
                'primary': primary,
                'Rprimary': 0.0,
            }
            return state, 0.0, False, False, info

        reward, info = get_reward(self.current_record, primary, self.location)

        self.prev_primary = primary
        self.prev_azimuth = 0.0

        state = self._build_state(self.current_record, primary)

        return state, reward, False, False, info

    def get_action_array(self, action_idx):
        return self.action_list[action_idx]