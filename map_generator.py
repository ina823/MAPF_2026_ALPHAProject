import numpy as np
from collections import deque
# ============================================================
# 1. 좌표 변환
# ============================================================

def rc_to_xy(rc):
    """
    내부 좌표 (row, col)를
    외부 CBS용 좌표 [x, y]로 변환한다.

    row = 세로 = y
    col = 가로 = x
    """
    row, col = rc
    return [int(col), int(row)]

# ============================================================
# 2. 가장 큰 연결된 빈 공간 찾기
# ============================================================

def largest_free_component(grid_map):
    """
    grid_map에서 가장 큰 연결된 빈 공간을 찾는다.

    0 = 빈 공간
    1 = 벽

    반환:
        [(row, col), ...]
    """

    rows, cols = grid_map.shape

    visited = np.zeros_like(grid_map, dtype=bool)

    largest_component = []

    for row in range(rows):
        for col in range(cols):

            if grid_map[row, col] != 0:
                continue

            if visited[row, col]:
                continue

            queue = deque([(row, col)])
            visited[row, col] = True

            component = []

            while queue:
                current_row, current_col = queue.popleft()

                component.append(
                    (current_row, current_col)
                )

                for drow, dcol in [
                    (-1, 0),   # 위
                    (1, 0),    # 아래
                    (0, -1),   # 왼쪽
                    (0, 1),    # 오른쪽
                ]:

                    next_row = current_row + drow
                    next_col = current_col + dcol

                    if not (0 <= next_row < rows):
                        continue

                    if not (0 <= next_col < cols):
                        continue

                    if grid_map[next_row, next_col] != 0:
                        continue

                    if visited[next_row, next_col]:
                        continue

                    visited[next_row, next_col] = True
                    queue.append(
                        (next_row, next_col)
                    )

            if len(component) > len(largest_component):
                largest_component = component

    return largest_component
# ============================================================
# 3. Agent 시작점 / 목표점 배치
# ============================================================

def place_agents(grid_map, num_agents, rng):
    """
    가장 큰 연결 공간 안에서
    agent의 start와 goal을 겹치지 않게 뽑는다.
    """

    free_cells = largest_free_component(grid_map)

    # agent 1대당 start 1개 + goal 1개 필요
    required_cells = 2 * num_agents

    # 빈칸이 부족하면 배치 실패
    if len(free_cells) < required_cells:
        return None, None

    # 같은 칸을 중복 선택하지 않게 뽑기
    selected_indices = rng.choice(
        len(free_cells),
        size=required_cells,
        replace=False,
    )

    selected_cells = [
        free_cells[index]
        for index in selected_indices
    ]

    starts = selected_cells[:num_agents]
    goals = selected_cells[num_agents:]

    return starts, goals
# ============================================================
# 4. 맵 종류 생성
# ============================================================

def _layout_empty(size, rng):
    """
    벽이 하나도 없는 맵
    """
    return np.zeros((size, size), dtype=int)


def _layout_random(size, rng, obstacle_probability):
    """
    랜덤하게 벽을 배치하는 맵

    obstacle_probability:
        각 칸이 벽이 될 확률
    """
    return (
        rng.random((size, size)) < obstacle_probability
    ).astype(int)


def _layout_rooms(size, rng, doors_per_wall=3):
    """
    방과 문 구조를 가진 맵
    """

    grid = np.zeros((size, size), dtype=int)

    wall_lines = [
        size // 3,
        2 * size // 3,
    ]

    # 세로벽 + 가로벽 만들기
    for line in wall_lines:
        grid[:, line] = 1
        grid[line, :] = 1

    # 벽에 문 뚫기
    for line in wall_lines:
        for _ in range(doors_per_wall):

            grid[
                rng.integers(0, size),
                line
            ] = 0

            grid[
                line,
                rng.integers(0, size)
            ] = 0

    return grid
def _layout_maze(size, rng):
    """
    미로 형태의 맵을 만든다.
    """

    # 처음에는 모든 칸을 벽(1)으로 만든다.
    grid = np.ones((size, size), dtype=int)

    # 시작점 하나를 빈칸으로 연다.
    grid[0, 0] = 0

    stack = [(0, 0)]

    while stack:

        row, col = stack[-1]

        candidates = []

        # 2칸씩 상 / 하 / 좌 / 우 확인
        for drow, dcol in [
            (-2, 0),
            (2, 0),
            (0, -2),
            (0, 2),
        ]:

            next_row = row + drow
            next_col = col + dcol

            if (
                0 <= next_row < size
                and 0 <= next_col < size
                and grid[next_row, next_col] == 1
            ):

                # 현재 칸과 다음 칸 사이에 있는 벽
                wall_row = row + drow // 2
                wall_col = col + dcol // 2

                candidates.append(
                    (
                        next_row,
                        next_col,
                        wall_row,
                        wall_col,
                    )
                )

        if candidates:

            index = rng.integers(len(candidates))

            (
                next_row,
                next_col,
                wall_row,
                wall_col,
            ) = candidates[index]

            # 다음 칸을 길로 만들기
            grid[next_row, next_col] = 0

            # 중간 벽도 뚫기
            grid[wall_row, wall_col] = 0

            stack.append(
                (next_row, next_col)
            )

        else:
            # 더 갈 곳이 없으면 이전 위치로 돌아감
            stack.pop()

    return grid