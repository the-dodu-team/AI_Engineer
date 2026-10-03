# 추가 탐지 모델 5종 실행

기존 10종에 RF-DETR Large, D-FINE-L(Objects365→COCO E25), LW-DETR Large, RF-DETR Medium, YOLOv13-L을 추가합니다. 사용자 데이터로 학습하지 않고 공개 가중치를 그대로 평가합니다.

## 실행

기존 `.venv`와 `requirements.txt` 환경에서 다음 명령을 실행합니다.

```powershell
.venv\Scripts\python.exe -m pip install -r requirements-extended.txt
.venv\Scripts\python.exe scripts\evaluate_extended_candidates.py --source 'G:\내 드라이브\side_project\sample'
.venv\Scripts\python.exe scripts\evaluate_extended_candidates.py --source 'G:\내 드라이브\side_project\sample' --models rtdetrv2-r50 rtdetrv2-r34 rtdetr-r50 rtdetr-r18 --output outputs/coco_corrected_rtdetr
.venv\Scripts\python.exe scripts\compare_extended_candidates.py
```

- 기본 출력: `outputs/coco_extended_detectors/`
- 비교 보고서: [reports/coco-extended-detectors/README.md](reports/coco-extended-detectors/README.md)
- 원본 파일의 SHA-256·정답 집합이 `outputs/rtmdet/model-size-20260927/run.json`과 다르면 추론 전에 중단합니다. 다른 위치의 기록은 `--baseline`으로 지정합니다.
- 모델별 별도 프로세스를 사용합니다. YOLOv13은 지정된 공식 소스 리비전을 내려받아 해당 프로세스에서만 사용합니다.
- 기본적으로 완료한 모델의 다운로드 캐시를 제거합니다. 가중치를 보관하려면 `--keep-weights`를 붙입니다. 가중치 해시와 모델 저장소 리비전은 결과에 남습니다.
- 실패한 모델이 있어도 다음 모델을 평가하고 `status.json`에 기록합니다. 같은 명령에 `--resume`을 붙이면 완료한 모델의 프로토콜·이미지 목록·예측 파일 해시를 검증한 뒤 재사용합니다.
- `--models rf-detr-large rf-detr-medium`처럼 일부 모델만 선택할 수 있습니다. 전체 비교 보고서는 5개 추가 모델과 기존 결과가 모두 완료되어야 생성됩니다.
- 기존 10종 및 RTMDet 예측 CSV를 다시 집계하며 이미지 누락·중복·정답 불일치가 있으면 보고서 생성을 중단합니다.

기존 RT-DETR 4종의 체크포인트가 `tv` 대신 `tvmonitor`를 반환하는데, 이전 스크립트는 이를 `other`로 매핑했습니다. 이 오류를 수정하고 `outputs/coco_corrected_rtdetr/`에 재평가합니다. 통합 보고서는 수정된 4종의 결과를 사용하며, 원래 파일은 보존합니다.

## 판정과 시간 측정

`laptop`·`tv`를 `computer`, `book`을 `book`, 나머지를 `other`로 매핑합니다. D-FINE의 `tvmonitor`도 같은 `tv`로 처리합니다. 각 그룹 최고 점수가 0.05 이상이면 후보가 되고, 가장 높은 그룹 하나를 고릅니다. 후보가 없으면 `other`입니다. 탐지 결과의 최소 confidence는 0.01입니다.

가중치가 사용하는 전처리를 유지합니다. RF-DETR Large는 704×704, Medium은 576×576이고 나머지는 640입니다. YOLOv13은 비율을 유지하는 letterbox를 사용합니다. 기본 CPU 10스레드, 배치 1, 워밍업 5회 후 이미지당 1회 측정합니다. `--repeats`로 반복을 늘릴 수 있습니다. 시간에는 전처리·추론·후처리가 포함되고 디스크 읽기와 모델 로딩은 제외됩니다.

기존 RTMDet는 다른 PyTorch/MMDetection 환경과 3회 반복을 사용했습니다. 보고서의 시간은 실행 환경별 참고값이며, 모델 간 속도 개선을 확정하는 통제된 비교는 아닙니다.

## 상업적 이용

| 모델 | 공개 조건 | 판단 |
|---|---|---|
| RF-DETR Large·Medium | 코드·공식 가중치 Apache-2.0 | 라이선스·저작권 고지 및 해당 NOTICE를 유지하는 조건으로 상업 이용 가능 |
| D-FINE-L Objects365→COCO | 코드·변환 모델 카드 Apache-2.0. 제작사가 Objects365 가중치의 추가 조건을 명시 | 상업 배포 전 가중치 권리 확인 필요 |
| LW-DETR Large | 코드·변환 모델 카드 Apache-2.0. Objects365 사전학습 사용 | 학습 데이터 조건이 가중치 이용에 미치는 범위 확인 필요 |
| YOLOv13-L | AGPL-3.0 | 상업 이용 자체가 금지되지는 않지만 배포·수정된 네트워크 서비스의 소스 제공 의무 등을 준수해야 함. 비공개 제품은 관련 권리자의 별도 허가 확인 필요 |

Objects365 사이트는 학술 목적 이용을 명시합니다. 데이터셋 조건만으로 모든 파생 가중치의 법적 상태를 단정하지 않고, D-FINE·LW-DETR을 상업 배포 확인이 필요한 후보로 표시합니다. Ultralytics Enterprise 라이선스가 외부 개발자의 YOLOv13 포크까지 포함한다고 가정하지 않습니다.

확인 출처: [RF-DETR](https://github.com/roboflow/rf-detr), [D-FINE](https://github.com/Peterande/D-FINE#model-zoo), [LW-DETR](https://github.com/Atten4Vis/LW-DETR), [Objects365](https://www.objects365.org/download.html), [YOLOv13 LICENSE](https://github.com/iMoonLab/yolov13/blob/main/LICENSE).

## 검증

```powershell
.venv\Scripts\python.exe -m pytest tests\test_extended_candidates.py -q
```

평가 환경: Python 3.12.10, CPU PyTorch 2.13.0, torchvision 0.28.0, Transformers 5.18.0. 모델별 실행 로그·전처리 설정·실제 파라미터 수·가중치 SHA-256은 `outputs/coco_extended_detectors/<모델>/`에 저장됩니다.
