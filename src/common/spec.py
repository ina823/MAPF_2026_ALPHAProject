"""
CBS-IL 공통 내부 표준 v0.3
이 파일은 표준에서 정의한 상수와 검증 로직만 담는다.
모델/데이터셋 코드가 전부 여기를 참조하므로, 표준이 v0.4로 바뀌면 여기만 고치면 된다.

v0.2 -> v0.3 변경점: local grid 채널 재정의. 빈공간 채널은 나머지 두 채널로부터
그대로 유도되는 중복 정보라 제거하고, 그 자리에 "다른 agent의 목표 위치" 채널을
추가했다 (PRIMAL류 관측 설계 참고). 내 목표는 이미 goal_dir로 정확한 상대좌표가
주어지므로 grid 채널에 다시 표시하지 않는다. 채널 수(GRID_C=3)와 npz shape은
그대로라 다운스트림 코드(dataset.py/model_*.py 등)는 변경이 필요 없다.
"""
import numpy as np

# ── 2. Action label ───────────────────────────────────────────────
# 0=상, 1=하, 2=좌, 3=우, 4=대기
ACTION_DELTA = {
    0: (-1, 0),   # 상
    1: ( 1, 0),   # 하
    2: ( 0, -1),  # 좌
    3: ( 0,  1),  # 우
    4: ( 0,  0),  # 대기
}
NUM_ACTIONS = 5
ACTION_NAMES = {0: "상", 1: "하", 2: "좌", 3: "우", 4: "대기"}

# ── 4. State 차원 ────────────────────────────────────────────────
GRID_C, GRID_H, GRID_W = 3, 5, 5          # (channel, row, col)
GRID_FLAT = GRID_C * GRID_H * GRID_W       # 75
GOAL_DIR_DIM = 2                           # [drow, dcol]
MLP_INPUT_DIM = GRID_FLAT + GOAL_DIR_DIM   # 77

# local grid 채널 정의 (v0.3). 한 칸은 아래 중 최대 하나만 1이고,
# 셋 다 0이면 빈공간이다 (빈공간은 별도 채널을 두지 않는다).
CH_WALL        = 0   # 벽 / 장애물 / 맵 밖
CH_OTHER_ROBOT = 1   # 다른 로봇의 현재 위치
CH_OTHER_GOAL  = 2   # 다른 agent의 목표 위치 (내 목표는 goal_dir로 별도 제공되므로 미포함)

# ── 6. .npz key ──────────────────────────────────────────────────
KEY_GRID = "states_grid"   # (N, 3, 5, 5) — 채널 정의는 위 CH_* 참고
KEY_GOAL = "goal_dirs"     # (N, 2)
KEY_ACT  = "actions"       # (N,)


def delta_to_action(drow: int, dcol: int) -> int:
    """이동 변위 (drow, dcol)을 action label로 역변환. CBS path → action 라벨링용."""
    for a, (dr, dc) in ACTION_DELTA.items():
        if (dr, dc) == (drow, dcol):
            return a
    raise ValueError(f"인접하지 않은 이동: ({drow}, {dcol})")


def path_to_actions(path):
    """
    3. CBS path(시간순 위치 목록)를 action 시퀀스로 변환.
    예: [(0,0),(0,1),(0,2)] -> [3, 3]  (우, 우)
    Track 1에서 CBS path 받으면 이 함수로 actions를 뽑으면 된다.
    """
    actions = []
    for (r0, c0), (r1, c1) in zip(path[:-1], path[1:]):
        actions.append(delta_to_action(r1 - r0, c1 - c0))
    return actions


def validate_npz(data):
    """로드한 npz가 v0.3 표준 shape을 지키는지 검증."""
    g, gd, a = data[KEY_GRID], data[KEY_GOAL], data[KEY_ACT]
    n = g.shape[0]
    assert g.shape == (n, GRID_C, GRID_H, GRID_W), f"grid shape 위반: {g.shape}"
    assert gd.shape == (n, GOAL_DIR_DIM), f"goal_dirs shape 위반: {gd.shape}"
    assert a.shape == (n,), f"actions shape 위반: {a.shape}"
    assert a.min() >= 0 and a.max() < NUM_ACTIONS, "action label 범위 위반"
    return n


def make_dummy_npz(path: str, n: int = 2000, seed: int = 0):
    """
    GT/시뮬레이터 없이 파이프라인을 검증하기 위한 더미 데이터.
    실제 학습용이 아니라 'forward/backward가 도는지' 확인용이다.
    의미 있는 신호를 약간 주려고, goal_dir 부호에 느슨하게 상관된 action을 생성한다.
    """
    rng = np.random.default_rng(seed)

    # local grid: 빈공간/벽/다른로봇/다른agent목표 4개 카테고리 중 하나를 뽑고,
    # 빈공간은 채널을 두지 않으므로(전부 0) 나머지 세 카테고리만 CH_*에 표시한다.
    grids = np.zeros((n, GRID_C, GRID_H, GRID_W), dtype=np.float32)
    FREE, WALL, ROBOT, GOAL = 0, 1, 2, 3
    cell_type = rng.choice([FREE, WALL, ROBOT, GOAL],
                            size=(n, GRID_H, GRID_W), p=[0.65, 0.2, 0.1, 0.05])
    grids[:, CH_WALL]        = (cell_type == WALL).astype(np.float32)
    grids[:, CH_OTHER_ROBOT] = (cell_type == ROBOT).astype(np.float32)
    grids[:, CH_OTHER_GOAL]  = (cell_type == GOAL).astype(np.float32)

    # 5. 목표방향 [goal_row-cur, goal_col-cur]. 임의 정수 변위.
    goal_dirs = rng.integers(-8, 9, size=(n, GOAL_DIR_DIM)).astype(np.float32)

    # 더미 라벨: 목표방향이 더 큰 축으로 이동하는 약한 규칙 + 노이즈
    actions = np.empty(n, dtype=np.int64)
    for i, (dr, dc) in enumerate(goal_dirs):
        if rng.random() < 0.15:
            actions[i] = rng.integers(0, NUM_ACTIONS)   # 노이즈
        elif abs(dr) >= abs(dc):
            actions[i] = 0 if dr < 0 else (1 if dr > 0 else 4)
        else:
            actions[i] = 2 if dc < 0 else 3

    np.savez(path, **{KEY_GRID: grids, KEY_GOAL: goal_dirs, KEY_ACT: actions})
    return n


if __name__ == "__main__":
    n = make_dummy_npz("dummy_v02.npz", n=2000)
    d = np.load("dummy_v02.npz")
    print(f"더미 {n}개 생성, 검증 통과: N={validate_npz(d)}")
    print("path_to_actions 테스트:", path_to_actions([(0, 0), (0, 1), (1, 1), (1, 1)]))
