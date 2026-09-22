# Alpha 1 -> Alpha 2 migration map (baseline migration)

Alpha 1 원본: https://github.com/qwedus/MAPF_2026_AlphaProject (조사 시점 2026-09-21)
로컬 참조본 `Alpha1_legacy` 는 **읽기 전용**이며 어떤 방식으로도 수정하지 않았다.

## 원칙

1. Alpha 1 IL-navhint baseline 의 동작을 바꾸지 않고 재현한다.
2. 원본은 branch 별 GitHub blob(LF)에서 그대로 복사한다. 바꾼 줄은 아래 "수정한 줄" 에 전부 적는다.
3. 새 알고리즘, fallback, heuristic 을 넣지 않는다. 새 코드는 Alpha 1 코드의 조립(`policy.py`), 실행/검증(`run_il.py`, `smoke_test_il.py`, tests)뿐이다.
4. Deadlock / Wait-for graph / Hybrid / CBS 관련 코드는 이번 단계에 없다.

## 1. 복사한 파일

`git hash-object <Alpha2 파일>` 이 원본 blob 과 같으면 verbatim 이다 (`.gitattributes` 로 LF 유지).

| Alpha 1 원본 | branch | 원본 blob | 마지막 커밋 | Alpha 2 | Alpha 2 blob | 상태 |
|---|---|---|---|---|---|---|
| `src/nav_hint.py` | analysis (= IL) | `97135d7d6777` | 7048e5e 2026-07-10 | `src/il/nav_hint.py` | `97135d7d6777` | verbatim |
| `src/model_navhint.py` | analysis (= IL) | `1d367fef4ba6` | 7048e5e 2026-07-10 | `src/il/model_navhint.py` | `1d7396524f01` | 1줄 수정 |
| `spec.py` | IL 만 | `e1951a275b3e` | 6dc6e41 2026-07-02 | `src/common/spec.py` | `e1951a275b3e` | verbatim |
| `simulator.py` | feature/simulator 만 | `c323d7d5f95d` | 8c4ac1a 2026-07-06 | `src/simulator/mapf_step_simulator.py` | `c323d7d5f95d` | verbatim |
| `src/scenario_loader.py` | main (= analysis, cbs-adapter) | `e586e71d05c3` | 0f91413 2026-07-01 | `src/common/scenario_loader.py` | `ee2c1d33bd39` | 1줄 수정 |
| `scenarios_dev/scenario_empty_s11_n3.json` | main | `02f232a444e2` | 423913e 2026-07-02 | `scenarios/smoke/scenario_empty_s11_n3.json` | `d2cd04b660bf` | 1줄 수정 |
| `scenarios_dev/map_empty_s11_n3.npy` | main | `d380c1e362c8` | 423913e 2026-07-02 | `scenarios/smoke/map_empty_s11_n3.npy` | `d380c1e362c8` | verbatim (binary) |
| `cnn_navhint.pt` | analysis 만 | `e3797a553ab1` | 7048e5e 2026-07-10 | `checkpoints/cnn_navhint.pt` | `e3797a553ab1` | verbatim (binary, sha256 은 MANIFEST.json) |

`simulator.py` 는 통째로 복사했다 (batch `Simulator` 와 `MAPFStepSimulator` 를 분리하지 않음). 파일 이름만 Alpha 2 구조에 맞췄다.

## 2. 수정한 줄 (3곳)

| 파일 | 변경 | 이유 |
|---|---|---|
| `src/il/model_navhint.py` 13행 | `import spec` -> `from src.common import spec` | `spec.py` 가 `src/common/` 으로 이동 (Alpha 1 에서는 루트 모듈) |
| `src/common/scenario_loader.py` 18행 | `parents[1]` -> `parents[2]` | 파일이 저장소 루트에서 두 단계 아래(`src/common/`)로 이동. `map_file` 이 `PROJECT_ROOT` 기준으로 해석되기 때문 |
| `scenarios/smoke/scenario_empty_s11_n3.json` 4행 | `"scenarios_dev/map_empty_s11_n3.npy"` -> `"map_empty_s11_n3.npy"` | 로더가 JSON 과 같은 폴더도 후보로 탐색하므로 맵을 JSON 옆에 둠 |

## 3. 새로 작성한 파일

| 파일 | 내용 |
|---|---|
| `src/il/policy.py` | thin wrapper. `predict` 는 `scripts/eval_cbs_vs_il.py::predict` verbatim, `NavHintPolicy` 는 `scripts/eval_flow.py::rollout` 의 BFS 캐시와 goal_dir overwrite 를 그대로 조립. 새 로직 없음 |
| `scripts/run_il.py` | `eval_flow.py::rollout` 의 episode loop 를 실행하는 CLI (`run_episode`) |
| `scripts/smoke_test_il.py` | baseline 재현 확인 (docs/alpha1_baseline.md 4절 기준) |
| `tests/test_simulator_characterization.py` | simulator 의 현재 동작 고정 (KI-1 포함) |
| `tests/test_nav_hint.py`, `tests/test_conventions.py`, `tests/test_smoke_il.py` | BFS/flowdist, action/좌표 규약, smoke 실행 |
| `tests/baselines/alpha1_il_navhint_reference.json` | 관측 reference (soft metadata) |
| `checkpoints/MANIFEST.json` | checkpoint sha256 / 출처 / 정규화 통계 |
| `__init__.py` (`src`, `src/il`, `src/simulator`, `src/common`, `scripts`, `tests`) | 빈 package 표식 |
| `README.md`, `requirements.txt`, `pytest.ini`, `.gitignore`, `.gitattributes`, `docs/*`, `configs/.gitkeep`, `outputs/*/.gitkeep` | 프로젝트 설정/문서 |

Alpha 2 에 원래 있던 `map_generator.py` 는 수정하지 않았다 (Alpha 1 의 어느 branch 버전과도 다른 별도 구현이고 scenario JSON 저장 함수가 없다).
`.gitignore` 는 기존 파일이 UTF-16(BOM) 6바이트로 사실상 빈 파일이어서 UTF-8 로 교체했다.

## 4. 가져오지 않은 Alpha 1 파일

| 분류 | 파일 | 이유 |
|---|---|---|
| 학습 | `scripts/train_navhint.py`, `train_dagger_navhint.py`, `build_flow_dataset.py`, IL 의 `dagger.py`, `dataset.py`, `train.py`, `eval.py`, `infer.py` | baseline 추론/rollout 에 불필요. DAgger/IL 재학습 금지 |
| 구 모델 | `model_cnn.py`, `model_mlp.py` | `eval_cbs_vs_il.py` 가 import 할 뿐 NavHintCNN 은 사용하지 않음 |
| 다른 checkpoint | `cnn.pt`, `cnn_big*.pt`, `cnn_diverse.pt`, `cnn_flow*.pt`, `cnn_navhint_both.pt`, `cnn_dagger_*.pt`, `mlp*.pt` | baseline 은 `cnn_navhint.pt` 하나 |
| 데이터 | `real_v03*.npz` (파일당 약 14 MB) | 학습/평가용 |
| 평가/분석 | `scripts/eval_*`, `plot_*`, `docs/img/`, `docs/*.md` | 참고 자료. 필요하면 Alpha 1 branch 를 링크 |
| CBS | `src/cbs_adapter.py`, `third_party/multi_agent_path_planning`, `scripts/run_cbs_batch.py` 등 | CBS 는 이번 범위 밖 (다음 단계에서 결정) |
| 기타 | `main.py`, `map_ex.py`, `examples/`, `outputs/`, `.venv-alpha1` | baseline 과 무관 |

## 5. 검증 방법

```powershell
# 텍스트 파일: 원본 blob 과 같으면 verbatim (예: nav_hint.py 는 97135d7d6777...)
git hash-object src/il/nav_hint.py
# 바이너리: sha256 이 checkpoints/MANIFEST.json 의 값과 같아야 함
(Get-FileHash checkpoints/cnn_navhint.pt -Algorithm SHA256).Hash
# baseline 재현
python -m scripts.smoke_test_il
python -m unittest discover -s tests -t . -v
```

Alpha 1 원본 blob 은 `git rev-parse origin/<branch>:<path>` 로 확인할 수 있다 (읽기 전용 조회).
