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
TAU = 0.005

# Stage 1
EPISODES = 500
MAX_STEP = 50
ANCHOR_DROP_PROB = 0.0
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 10000
LR = 1e-4

# Stage 2 (fine-tuning)
FINETUNE_EPISODES = 300
FINETUNE_DROP_PROB = 0.2
FINETUNE_EPS_START = 0.3
FINETUNE_LR = 5e-5

# 2-phase: acquisition → tracking
CONVERGENCE_VAR_THRESHOLD = 4.0   # m²: 최근 추정값 분산이 이 값 미만이면 수렴 판단
RECENT_EST_WINDOW = 5             # 분산 계산에 사용할 최근 추정값 개수
