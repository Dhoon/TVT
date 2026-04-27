ANCHOR_POSITIONS = {
    1: (-0.75, 1.50),
    2: ( 0.75, 1.50),
    3: (-0.75, 0.00),
    4: ( 0.75, 0.00),
    5: (-0.57, -0.30),
    6: ( 0.57, -0.30),
}


C = 299702547
DWT_TIME_UNIT = 1 / (499.2e6 * 128.0)

BATCH_SIZE = 64
GAMMA = 0.9
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 800
TAU = 0.005
LR = 1e-4
EPISODES = 1000
MAX_STEP = 50
