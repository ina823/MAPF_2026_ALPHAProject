# MAPF_2026_ALPHAProject (Alpha 2)

Deadlock-Aware Hybrid MAPF with Selective CBS Replanning and Real-Robot Validation
(교착 감지 및 선택적 CBS 재계획 기반 하이브리드 MAPF 및 실물 검증)

## 현재 단계: Alpha 1 IL-navhint baseline 재현 (migration)

이 저장소에는 지금 **Alpha 1 의 IL-navhint baseline 과 `MAPFStepSimulator`** 만 옮겨져 있다.
Deadlock Monitor, Wait-for graph, Hybrid(Global/Selective) 재계획, CBS 변경, 재학습, 실물 로봇 코드는 아직 없다.

```
IL-navhint policy (cnn_navhint.pt, NavHintCNN, flowdist)
  -> MAPFStepSimulator (Python + NumPy grid-world)
```

## 구조

```
checkpoints/   cnn_navhint.pt, MANIFEST.json (sha256 / 출처 / 정규화 통계)
scenarios/     smoke/ (Alpha 1 scenario_empty_s11_n3)
src/common/    spec.py, scenario_loader.py
src/il/        model_navhint.py, nav_hint.py, policy.py
src/simulator/ mapf_step_simulator.py
scripts/       smoke_test_il.py, run_il.py
tests/         단위/특성 테스트, baselines/ (reference 관측값)
docs/          alpha1_baseline.md, migration_map.md
outputs/       logs, csv, figures, videos (git 에는 폴더만)
map_generator.py   Alpha 2 자체 맵 생성기 (이번 migration 에서 수정하지 않음)
```

## 설치 (Windows, Python 3.14 에서 검증)

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

## 실행

저장소 루트에서 실행한다.

```powershell
python -m scripts.smoke_test_il                    # baseline 재현 확인 (torch 필요)
python -m scripts.smoke_test_il --strict-baseline  # reference 관측값과 어긋나면 FAIL
python -m scripts.run_il --verbose                 # 시나리오 1개 rollout
python -m unittest discover -s tests -t . -v       # 테스트 (pytest 로도 실행 가능)
```

## 문서

- [docs/alpha1_baseline.md](docs/alpha1_baseline.md): baseline 정의, 규약, reference 결과, known issue, provenance
- [docs/migration_map.md](docs/migration_map.md): Alpha 1 원본 -> Alpha 2 파일 매핑, 수정한 줄, 가져오지 않은 파일

Alpha 1 원본 저장소(https://github.com/qwedus/MAPF_2026_AlphaProject)는 참조용이며 이 저장소에서 수정하지 않는다.
