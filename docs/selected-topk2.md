# 선택 모델 Top-2 실험

대상 모델은 아래 3종으로 고정합니다.

- RT-DETRv2-R50
- RT-DETRv2-R34
- LW-DETR Large

이 실험은 fine-tuning이 아니라 COCO 사전학습 가중치의 추론 후처리 비교입니다. 동일한 추론에서 저장한 `computer`, `book`, `other` 그룹 점수로 Top-1과 Top-2를 동시에 계산하므로 k별 추론 편차가 없습니다.

## 환경 준비

저장소 루트의 PowerShell에서 실행합니다.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-extended.txt
```

## 전체 실행

`--source`에는 `computer/` 또는 `laptop/`, `book/`, 기타 비대상 폴더가 들어 있는 기존 372장 데이터 루트를 지정합니다.

```powershell
.\.venv\Scripts\python.exe scripts\run_selected_topk.py --source "G:\내 드라이브\side_project\sample" --device cpu
```

GPU를 사용하려면 `--device cuda:0`으로 바꿉니다. 기본 출력은 `outputs/selected-topk2`, 보고서는 `docs/reports/selected-topk2`입니다.

중단된 실행을 이어서 진행하려면 동일한 옵션에 `--resume`을 추가합니다.

```powershell
.\.venv\Scripts\python.exe scripts\run_selected_topk.py --source "G:\내 드라이브\side_project\sample" --device cpu --resume
```

추론 결과가 이미 있고 보고서만 다시 만들려면:

```powershell
.\.venv\Scripts\python.exe scripts\run_selected_topk.py --report-only
```

실행기는 데이터의 SHA-256 집합이 기존 RTMDet 372장 manifest와 정확히 일치하는지 먼저 검증합니다. 완료 후 다음을 생성합니다.

- 선택 모델별 Top-1·Top-2 지표와 이미지별 후보
- Top-2로 추가 복구된 정답 수
- Top-1 대비 FNR·FPR 변화
- RTMDet-x Top-1 결과가 포함된 통합 비교표

Top-2 정답 포함률은 단일 분류 정확도가 아닙니다. 자동 승인 정책을 검토할 때는 FNR뿐 아니라 FPR, 클래스별 precision, 복수 후보율을 함께 확인해야 합니다.
