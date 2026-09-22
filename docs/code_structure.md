\# MAPF Alpha 2 코드 구조 정리



> 프로젝트: Deadlock-Aware Hybrid MAPF with Selective CBS Replanning  

> 현재 단계: Alpha1 IL-navhint baseline 이식 및 Simulator / Common Logging 구축  

> 현재 구현 범위에서는 Deadlock Detector, Wait-for Graph, Global/Selective Hybrid는 아직 구현하지 않음.



\---



\## 1. 전체 실행 흐름



현재 IL baseline의 기본 실행 흐름은 다음과 같다.



Scenario

↓

scripts/run\_il.py

↓

NavHintPolicy

↓

Observation + BFS flowdist navhint

↓

IL Model (NavHintCNN)

↓

Action 선택

↓

MAPFStepSimulator.step()

↓

이동 / 충돌 처리 / 위치 갱신

↓

EpisodeLogger

↓

CSV Logging



즉,



1\. Scenario에서 map, start, goal을 불러온다.

2\. IL Policy가 각 agent의 행동을 선택한다.

3\. Simulator가 실제 timestep을 진행한다.

4\. 충돌 규칙에 따라 최종 위치가 결정된다.

5\. Logging을 활성화한 경우 timestep별 결과를 CSV에 저장한다.



\---



\# 2. 주요 폴더



\## `scripts/`



실험이나 simulation을 직접 실행하는 entry point가 위치한다.



현재 주요 파일:



\- `run\_il.py`

\- `smoke\_test\_il.py`



\---



\## `src/common/`



여러 알고리즘에서 공통으로 사용할 코드가 위치한다.



현재 주요 파일:



\- `spec.py`

\- `scenario\_loader.py`

\- `episode\_logger.py`



향후 IL / CBS / Hybrid가 동일한 형식으로 결과를 기록하도록 공통 기능을 이곳에 둘 수 있다.



\---



\## `src/il/`



Alpha1에서 가져온 IL-navhint baseline 관련 코드가 위치한다.



현재 주요 파일:



\- `model\_navhint.py`

\- `nav\_hint.py`

\- `policy.py`



\---



\## `src/simulator/`



MAPF simulation 환경이 위치한다.



현재 핵심 파일:



\- `mapf\_step\_simulator.py`



Agent의 위치 이동, collision 처리, timestep 진행 등의 핵심 환경 동작을 담당한다.



\---



\## `scenarios/`



simulation에 사용할 map / start / goal scenario를 저장한다.



현재 smoke test용 scenario가 포함되어 있다.



\---



\## `checkpoints/`



IL 학습 완료 모델 checkpoint를 저장한다.



현재 주요 파일:



\- `cnn\_navhint.pt`

\- `MANIFEST.json`



\---



\## `outputs/`



실험 결과를 저장한다.



예:



\- `outputs/logs/`

\- `outputs/csv/`

\- `outputs/figures/`

\- `outputs/videos/`



생성된 실험 결과 파일 자체는 기본적으로 Git에 올리지 않고,

폴더 구조 유지를 위한 `.gitkeep`만 관리한다.



\---



\## `tests/`



baseline 및 새 기능이 기존 동작을 깨뜨리지 않는지 자동 검증한다.



\---



\# 3. 핵심 파일 역할



\## `scripts/run\_il.py`



\### 역할



IL-navhint simulation의 실행 entry point.



\### 주요 흐름



1\. Scenario load

2\. IL checkpoint load

3\. `NavHintPolicy` 생성

4\. Simulator reset

5\. 매 timestep마다 policy action 결정

6\. `sim.step(actions)` 실행

7\. episode 종료까지 반복

8\. 필요하면 CSV logging 수행



\### Logging



`--log` 옵션을 사용하면 `EpisodeLogger`가 활성화된다.



예:



```bash

python -m scripts.run\_il --log

