# MAPF_2026_ALPHAProject_2 (Alpha 2)

**Deadlock-Aware Hybrid MAPF with Selective CBS Replanning and Real-Robot Validation**

교착 감지 및 선택적 CBS 재계획 기반 하이브리드 MAPF 및 실물 검증

## 1. 현재 상태

Alpha 2는 Alpha 1의 결과를 그대로 덮어쓰는 프로젝트가 아니라, **재현 가능한 IL-navhint baseline을 고정한 뒤 deadlock/stagnation을 측정하고 Hybrid MAPF로 확장하는 후속 연구**이다.

현재 이 저장소에서 구현·검증된 범위는 다음과 같다.

| 구분 | 상태 | 현재 근거 |
| --- | --- | --- |
| Alpha 1 IL-navhint baseline 이관 | ✅ 완료 | `cnn_navhint.pt`, NavHintCNN, flowdist, smoke test |
| MAPFStepSimulator 재현 | ✅ 완료 | wall / vertex / edge collision, WAIT, timeout |
| Common Logging | ✅ 완료 | step-level / episode-level CSV |
| Offline Failure Analysis | ✅ 완료 | WAIT, blocked, no-progress, oscillation, joint-state repeat 등 |
| Controlled scenarios 4종 | ✅ 완료 | corridor / intersection / bottleneck / cycle |
| CBS current-state replanning 기반 | 🟡 팀 CBS 파트에서 검증, Alpha 2 통합 전 | `replan(current_positions, goals, grid_map)` interface |
| Runtime Deadlock Monitor | ⏳ 예정 | threshold/definition 검증 후 구현 |
| Wait-for Graph | ⏳ 예정 | affected-agent group 추출 |
| Global / Selective Hybrid | ⏳ 예정 | CBS 통합 후 구현 |
| Real-Robot Validation | ⏳ 예정 | Simulation Core 완료 후 진행 |

> **중요:** 현재 controlled scenario에서 관찰한 `stagnation_candidate`는 최종 deadlock label이 아니다. Deadlock 정의와 threshold는 validation 단계에서 확정한다.
> 

---

## 2. 저장소와 브랜치

- Alpha 2 현재 원격 저장소: `ina823/MAPF_2026_ALPHAProject_2`
- 기본 브랜치: `main`
- 현재 개발 브랜치: `feature/common-logging`
- Alpha 1 원본: `qwedus/MAPF_2026_AlphaProject` — **read-only reference**

Alpha 1 원본은 기존 결과와 provenance를 보존하기 위해 수정하지 않는다.

Alpha 2에서 재현·계측·Hybrid 확장을 진행한다.

---

## 3. 현재 Alpha 2 IL 실행 Pipeline

```
Scenario JSON + Grid Map
        ↓
src/common/scenario_loader.py
        ↓
scripts/run_il.py
        ↓
NavHintPolicy
        ↓
Local Observation (3×5×5)
+ BFS flowdist navhint (2-D)
        ↓
NavHintCNN / cnn_navhint.pt
        ↓
UP / DOWN / LEFT / RIGHT / WAIT
        ↓
MAPFStepSimulator.step()
        ↓
wall / vertex / edge collision 처리
        ↓
next positions + episode state
        ↓
EpisodeLogger (--log)
        ↓
outputs/logs/*_steps.csv
outputs/logs/*_summary.csv
        ↓
scripts/analyze_failure_logs.py
        ↓
outputs/failure_analysis/*_events.csv
outputs/failure_analysis/*_episode.csv
```

### IL policy 입력

각 agent는 다음 정보를 사용한다.

1. **Local observation grid:** `(3, 5, 5)`
    - ch0: wall / obstacle / map outside
    - ch1: other robot current positions
    - ch2: other agent goals
2. **BFS flowdist navhint:** `(2,)`
    - 정적 장애물을 고려한 BFS distance field 기반 방향 정보

### Action convention

| ID | Action | Delta |
| --- | --- | --- |
| 0 | UP | (-1, 0) |
| 1 | DOWN | (1, 0) |
| 2 | LEFT | (0, -1) |
| 3 | RIGHT | (0, 1) |
| 4 | WAIT | (0, 0) |

---

## 4. CBS / Expert Path / Simulator 흐름

Alpha 1 및 팀 CBS 파트에서 확인한 CBS 흐름은 다음과 같다.

```
Map / Start / Goal
        ↓
map_generator.py
        ↓
src/cbs_adapter.py
CBSAdapter.plan()
        ↓
CBS input.yaml
        ↓
third_party/multi_agent_path_planning/
centralized/cbs/cbs.py
        ↓
CBS output.yaml
        ↓
CBSAdapter 좌표 복원
        ↓
{agent_id: [(row, col), ...]}
        ↓
Simulator path validation
        ↓
collision-free expert path
```

Alpha 1에서는 이와 같은 CBS 경로를 expert trajectory로 활용해 Behavioral Cloning 기반 IL-navhint 정책을 구성했다.

**Alpha 2에서는 기존 `cnn_navhint.pt`를 baseline으로 고정하며, 현재 단계에서 IL 재학습은 수행하지 않는다.**

---

## 5. CBS ↔︎ 프로젝트 좌표 규약

프로젝트 내부와 atb033 CBS의 좌표 순서가 다르므로 반드시 Adapter를 거친다.

```
Project
(row, col)

    ↓ CBSAdapter

CBS
[x, y] = [col, row]

    ↓ CBSAdapter

Project
(row, col)
```

예:

```
Project: (row=5, col=2)
CBS:     [x=2, y=5]
```

CBS 원본 출력은 `t, x, y` schedule이지만, Adapter 반환 후에는 timestep을 list index로 표현한다.

```python
{
    0: [(3, 2), (3, 3), (3, 4)],
    1: [(4, 4), (4, 4), (4, 4)]
}
```

즉 `path[0]`이 replan 내부 `t=0`, `path[1]`이 `t=1`이다.

---

## 6. Current-State Replanning Integration Contract

팀 CBS 파트에서 다음 인터페이스를 검증했다.

```python
replan(
    current_positions,
    goals,
    grid_map,
    timeout_sec=30,
)
```

### 입력

| 입력 | 의미 | 형식 |
| --- | --- | --- |
| `current_positions` | 현재 timestep의 agent 위치 | `[(row, col), ...]` |
| `goals` | 기존 최종 목적지 | `[(row, col), ...]` |
| `grid_map` | 현재 map | 2D grid |
| `timeout_sec` | CBS 최대 실행시간 | 기본 30초 |

### 출력

```python
{
    agent_id: [
        (row, col),
        (row, col),
        ...
    ]
}
```

반환 좌표는 이미 프로젝트 내부의 `(row, col)` 형식이므로 추가 x/y 변환 없이 Simulator 측과 연결할 수 있다.

### Global timestep과 replan timestep

예를 들어 episode의 `global t=20`에서 CBS를 호출해도 새 CBS path는 다시 `replan t=0`부터 시작한다.

```
global t=20  ↔ replan t=0
global t=21  ↔ replan t=1
global t=22  ↔ replan t=2
```

향후 Hybrid state machine에서는 두 timestep을 분리해 관리한다.

> **현재 integration boundary:** 위 CBS interface는 팀 CBS 파트에서 검증되었지만 아직 이 Alpha 2 저장소의 IL runtime과 end-to-end로 merge된 상태는 아니다. README는 통합 전 interface contract를 명시한다.
> 

---

## 7. 목표 Hybrid Pipeline

최종적으로 구현할 구조는 다음과 같다.

```
MAPF Scenario / Goal
        ↓
Current World State
        ↓
IL-navhint Policy
        ↓
MAPFStepSimulator
        ↓
Common Logging
        ↓
Deadlock Monitor
   ┌────┴────┐
정상       위험/교착
 ↓            ↓
IL 계속   Wait-for Graph
              ↓
       affected-agent group
          ┌────┴────┐
       Global     Selective
         CBS         CBS
          └────┬────┘
               ↓
          Recovery path
               ↓
        progress 재개 확인
               ↓
          IL mode 복귀
```

연구 핵심은 **Selective가 항상 더 좋다고 가정하는 것이 아니라**, Global CBS와 Selective CBS를 동일 조건에서 비교해 latency, recovery success, target-agent ratio 등의 trade-off를 데이터로 검증하는 것이다.

---

## 8. Common Logging

`src/common/episode_logger.py`는 현재 IL에 연결되어 있으며 향후 CBS / Hybrid에서도 재사용하도록 독립적으로 구성했다.

### Step-level CSV

주요 필드:

```
run_id
scenario
mode
timestep
agent_id
current_row/current_col
action_id/action_name
is_wait
intended_row/intended_col
next_row/next_col
goal_distance_before/after
progress
conflict_type
blocked
done
success
failure_reason
trigger_reason
cbs_latency_ms
```

- `is_wait=True`: policy가 실제 WAIT을 선택
- `blocked=True`: 이동하려 했지만 wall/conflict 때문에 움직이지 못함
- `trigger_reason`, `cbs_latency_ms`: 향후 Hybrid용 예약 필드

### Episode summary

```
run_id, scenario, mode, num_agents, steps, success, timed_out,
makespan, wait_count, blocked_count, vertex_conflicts, edge_conflicts
```

실제 생성되는 결과 CSV는 Git에 올리지 않고 `outputs/` 아래에서 관리한다.

---

## 9. Offline Failure Analysis

`scripts/analyze_failure_logs.py`는 rollout을 변경하지 않는 **offline evidence-analysis tool**이다.

현재 분석하는 주요 신호:

- vertex / edge conflict
- active WAIT
- blocked
- BFS no-progress
- goal-distance stagnation
- short oscillation
- joint-state repeat
- repeated conflict candidate

### 주의

- `stagnation_candidate`는 deadlock 최종 판정이 아니다.
- repeated conflict는 현재 event ID가 없으므로 reconstruction candidate로 취급한다.
- threshold는 아직 freeze하지 않는다.
- 성공한 episode는 stagnation/deadlock으로 분류하지 않는다.

---

## 10. Controlled Scenarios

`scenarios/controlled/`에는 다음 4종을 둔다.

| Scenario | Agents | 목적 |
| --- | --- | --- |
| corridor_bay_n2 | 2 | head-on + 양보 필요 |
| intersection_n2 | 2 | 중앙 cell 우선순위 조정 |
| bottleneck_n4 | 4 | 1-cell door 양방향 통행 |
| cycle_ring_n4 | 4 | coordination 성공 대조 사례 |

각 scenario는 hand-written joint plan으로 **해가 존재함을 별도 검증**하여, 애초에 unsolvable한 문제와 IL policy의 coordination failure를 구분한다.

현재 1차 관찰에서는 corridor / intersection / bottleneck에서 persistent vertex-conflict 기반 stagnation이 재현되었고, cycle ring은 성공했다.

이 결과만으로 전체 실패 중 deadlock 비율이나 최적 threshold를 주장하지 않는다.

---

## 11. Repository Structure

```
checkpoints/
  cnn_navhint.pt
  MANIFEST.json

configs/

docs/
  alpha1_baseline.md
  migration_map.md
  code_structure.md

scenarios/
  smoke/
  controlled/

scripts/
  smoke_test_il.py
  run_il.py
  analyze_failure_logs.py

src/
  common/
    spec.py
    scenario_loader.py
    episode_logger.py
  il/
    model_navhint.py
    nav_hint.py
    policy.py
  simulator/
    mapf_step_simulator.py

tests/
  baselines/
  test_conventions.py
  test_nav_hint.py
  test_simulator_characterization.py
  test_smoke_il.py
  test_episode_logger.py
  test_failure_analysis.py

outputs/
  logs/
  failure_analysis/
  csv/
  figures/
  videos/

map_generator.py
```

---

## 12. 설치

Windows / Python 환경 기준.

```powershell
python -m venv .venv
.venv\Scripts\activate

pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

---

## 13. 실행 및 검증

저장소 루트에서 실행한다.

### Baseline smoke test

```powershell
python -m scripts.smoke_test_il
python -m scripts.smoke_test_il --strict-baseline
```

### IL rollout

```powershell
python -m scripts.run_il --verbose
```

### Logging 포함 rollout

```powershell
python -m scripts.run_il --log --verbose
```

특정 controlled scenario 실행 예:

```powershell
python -m scripts.run_il --scenario scenarios/controlled/scenario_corridor_bay_n2.json --log --verbose
```

### Failure analysis

```powershell
python -m scripts.analyze_failure_logs outputs/logs
```

또는 특정 run만 분석:

```powershell
python -m scripts.analyze_failure_logs outputs/logs/<run_id>_steps.csv
```

### 전체 unit test

```powershell
python -m unittest discover -s tests -t . -v
```

---

## 14. Baseline Preservation Rules

Alpha 2 연구에서 다음 원칙을 유지한다.

1. Alpha 1 원본 저장소는 read-only reference로 유지한다.
2. 기존 IL baseline의 동작을 임의로 개선한 뒤 baseline이라고 부르지 않는다.
3. Logging / analysis 추가가 baseline trajectory를 바꾸지 않는지 계속 regression test한다.
4. Deadlock definition과 threshold는 validation 데이터로 정한 뒤 freeze한다.
5. test 결과를 본 뒤 threshold를 결과에 맞춰 변경하지 않는다.
6. Global / Selective CBS는 동일 experiment harness에서 비교한다.
7. 실물 로봇 결과는 feasibility evidence로 사용하고, 정량적 주장은 Simulation을 중심으로 한다.

---

## 15. 현재 다음 Integration Gate

현재 Alpha 2의 다음 공동 작업은 새로운 알고리즘을 무작정 추가하는 것이 아니라 아래 interface를 실제 runtime으로 연결하는 것이다.

```
IL rollout
  ↓
current_positions / goals / grid_map 추출
  ↓
replan(current_positions, goals, grid_map)
  ↓
CBS path
  ↓
Simulator / Hybrid execution
  ↓
Common Logging
```

그 다음 단계에서 Deadlock Monitor와 Wait-for Graph를 연결한다.

---

## 16. 관련 문서

- docs/alpha1_baseline.md — Alpha 1 baseline 정의와 provenance
- docs/migration_map.md — Alpha 1 → Alpha 2 파일 이관 관계
- docs/code_structure.md — 현재 IL / Simulator / Logging 코드 구조

Alpha 1 원본 저장소: https://github.com/qwedus/MAPF_2026_AlphaProject