import random

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
            for leaf in remaining:
                self.action_list.append([root, leaf])
        self.action_space = spaces.Discrete(len(self.action_list))

        # state: [azimuth, prev_primary] + 6×3 timestamps = 20
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(20,), dtype=np.float32
        )

        self.location = 1
        self.current_record = None
        self.prev_record = None
        self.prev_est = None
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

    def _build_state(self, record, prev_primary):
        messages = record.get('messages', {})

        UINT32 = 1 << 32
        otj = []

        for anchor_id in range(1, 7):
            if anchor_id == prev_primary:
                msg = messages.get(str(anchor_id), [])
                if len(msg) < 2:
                    otj.extend([-1.0, -1.0, -1.0])
                    continue
                Ra = float(msg[0])
                Da = float(msg[1])
                if Ra == 0 or Da == 0:
                    otj.extend([-1.0, -1.0, -1.0])
                    continue
                otj.extend([1.0, Ra + 1, Ra + Da + 1])
            else:
                msg = messages.get(str(anchor_id), [0, 0, 0])
                if len(msg) < 3:
                    msg = msg + [0] * (3 - len(msg))
                msg3 = [float(v) for v in msg[:3]]
                if all(v == 0 for v in msg3):
                    otj.extend([-1.0, -1.0, -1.0])
                    continue
                base_ts = next(v for v in msg3 if v != 0)
                for v in msg3:
                    if v == 0:
                        otj.append(-1.0)
                    else:
                        rel = (v - base_ts) % UINT32
                        normalized = (rel / 10000000.0) + 1.0
                        otj.append(normalized)

        state = [float(self.prev_azimuth), float(prev_primary)] + otj
        return np.array(state, dtype=np.float32)

    def reset(self, seed=None, options=None):
        self.prev_primary = random.randint(1, 6)
        self.current_record = self._get_record(self.location, self.prev_primary)

        if self.current_record is None:
            self.current_record = self._get_record(self.location)
        if self.current_record is None:
            return np.zeros(20, dtype=np.float32), {}

        leaf = random.choice([i for i in range(1, 7) if i != self.prev_primary])
        est, _, _ = estimate_position_for_action(self.current_record, self.prev_primary, leaf)
        self.prev_azimuth = calc_azimuth(est[0], est[1]) if est is not None else 0
        self.prev_est = tuple(est) if est is not None else None
        self.prev_record = self.current_record

        state = self._build_state(self.current_record, self.prev_primary)
        return state, {}

    def step(self, action_idx):
        global record_fail_count

        action = self.action_list[action_idx]
        primary = action[0]

        self.current_record = self._get_record(self.location, primary)
        if self.current_record is None:
            record_fail_count += 1
            if self.prev_record is not None:
                fail_state = self._build_state(self.prev_record, primary)
                fail_state[0] = -1.0
                anchor_offset = 2 + (primary - 1) * 3
                fail_state[anchor_offset:anchor_offset + 3] = -1.0
            else:
                fail_state = np.full(20, -1.0, dtype=np.float32)
                fail_state[1] = float(primary)
            return fail_state, -1.0, False, False, {'fail': True, 'record': None}

        est, _, pos_latency_ms = estimate_position_for_action(self.current_record, action[0], action[1], self.prev_est)

        self.prev_azimuth = calc_azimuth(est[0], est[1]) if est is not None else self.prev_azimuth
        self.prev_est = tuple(est) if est is not None else self.prev_est
        self.prev_primary = primary
        self.prev_record = self.current_record

        state = self._build_state(self.current_record, primary)
        reward, info = get_reward(self.current_record, action, self.location)

        info['estimated_position'] = [round(est[0], 4), round(est[1], 4)] if est is not None else None
        info['record'] = self.current_record
        info['fail'] = False
        info['pos_latency_ms'] = pos_latency_ms

        return state, reward, False, False, info

    def get_action_array(self, action_idx):
        return self.action_list[action_idx]
