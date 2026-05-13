import random
from itertools import combinations

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from data_loader import load_json_data
from reward import get_reward, GROUND_TRUTH
from utils import calc_azimuth, estimate_position_for_action

record_fail_count = 0


class CustomEnv(gym.Env):
    def __init__(self, script_dir=None):
        super(CustomEnv, self).__init__()

        self.data, self.pedloc_counts = load_json_data(script_dir)

        self.action_list = []
        for root in range(1, 7):
            remaining = [j for j in range(1, 7) if j != root]
            for combo in combinations(remaining, 2):
                self.action_list.append([root] + list(combo))
        self.action_space = spaces.Discrete(len(self.action_list))

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(26,), dtype=np.float32
        )

        self.location = 1
        self.current_record = None
        self.prev_azimuth = 0
        self.prev_primary = 1

    def _get_record(self, location, primary=None):
        records = self.data.get(location, [])
        valid = [r for r in records if r.get('messages')]
        if primary is not None:
            valid = [r for r in valid if r.get('root_anchor') == primary]
        if not valid:
            return None
        return random.choice(valid)

    def _build_state(self, record, prev_primary, noise_std=1000.0, augment=False):
        messages = record.get('messages', {})

        UINT32 = 1 << 32
        otj = []

        for anchor_id in range(1, 7):
            if anchor_id == prev_primary:
                # root: msg[1:5] (뒤 4개)
                msg = messages.get(str(anchor_id), [])
                if len(msg) < 5:
                    otj.extend([-1.0, -1.0, -1.0, -1.0])
                    continue
                msg4 = [float(v) for v in msg[1:5]]
                if all(v == 0 for v in msg4):
                    otj.extend([-1.0, -1.0, -1.0, -1.0])
                    continue
                base_ts = next(v for v in msg4 if v != 0)
                for v in msg4:
                    if v == 0:
                        otj.append(-1.0)
                    else:
                        rel = (v - base_ts) % UINT32
                        normalized = (rel / 10000000.0) + 1.0
                        if augment and normalized > 1.0:
                            normalized += np.random.normal(0, noise_std)
                        otj.append(normalized)
            else:
                # leaf: msg[0:4]
                msg = messages.get(str(anchor_id), [0, 0, 0, 0])
                if len(msg) < 4:
                    msg = msg + [0] * (4 - len(msg))
                msg4 = [float(v) for v in msg[:4]]
                if all(v == 0 for v in msg4):
                    otj.extend([-1.0, -1.0, -1.0, -1.0])
                    continue
                base_ts = next(v for v in msg4 if v != 0)
                for v in msg4:
                    if v == 0:
                        otj.append(-1.0)
                    else:
                        rel = (v - base_ts) % UINT32
                        normalized = (rel / 10000000.0) + 1.0
                        if augment and normalized > 1.0:
                            normalized += np.random.normal(0, noise_std)
                        otj.append(normalized)

        state = [float(self.prev_azimuth), float(prev_primary)] + otj
        return np.array(state, dtype=np.float32)

    def reset(self, seed=None, options=None):
        self.prev_primary = random.randint(1, 6)
        self.current_record = self._get_record(self.location, self.prev_primary)

        if self.current_record is None:
            self.current_record = self._get_record(self.location)
        if self.current_record is None:
            return np.zeros(26, dtype=np.float32), {}

        leaves = random.sample([i for i in range(1, 7) if i != self.prev_primary], 2)
        est, _ = estimate_position_for_action(self.current_record, self.prev_primary, leaves[0], leaves[1])
        self.prev_azimuth = calc_azimuth(est[0], est[1]) if est is not None else 0

        state = self._build_state(self.current_record, self.prev_primary)
        return state, {}

    def step(self, action_idx):
        global record_fail_count

        action = self.action_list[action_idx]
        primary = action[0]

        self.current_record = self._get_record(self.location, primary)
        if self.current_record is None:
            record_fail_count += 1
            return np.zeros(26, dtype=np.float32), 0.0, False, False, {'fail': True, 'record': None}

        est, _ = estimate_position_for_action(self.current_record, action[0], action[1], action[2])
        self.prev_azimuth = calc_azimuth(est[0], est[1]) if est is not None else 0
        self.prev_primary = primary

        state = self._build_state(self.current_record, primary)
        reward, info = get_reward(self.current_record, action, self.location)

        info['estimated_position'] = [round(est[0], 4), round(est[1], 4)] if est is not None else None
        info['record'] = self.current_record
        info['fail'] = False

        return state, reward, False, False, info

    def get_action_array(self, action_idx):
        return self.action_list[action_idx]