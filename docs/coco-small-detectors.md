# COCO 소형 탐지 모델 10종 평가

추가 5종의 실행 방법과 상업 이용 조건은 [추가 모델 평가](coco-extended-detectors.md), 총 20개 모델 비교는 [통합 결과](reports/coco-extended-detectors/README.md)를 확인하세요.

이 실험은 RTMDet-l(52.3M 파라미터)보다 작은 COCO 사전학습 탐지 모델을 같은 이미지 수준 판정으로 비교합니다. 공개 COCO AP는 후보 선정 자료이며 이 프로젝트의 FPR·FNR을 예측하지 않습니다.

| 우선순위 | 모델 | 파라미터(M) | 공개 COCO AP | 백엔드 |
|---:|---|---:|---:|---|
| 1 | YOLO26l | 24.8 | 55.0 | Ultralytics |
| 2 | YOLO26m | 20.4 | 53.1 | Ultralytics |
| 3 | YOLO11l | 25.3 | 53.4 | Ultralytics |
| 4 | YOLO11m | 20.1 | 51.5 | Ultralytics |
| 5 | YOLOv10l | 24.4 | 53.2 | Ultralytics |
| 6 | YOLOv10m | 15.4 | 51.1 | Ultralytics |
| 7 | RT-DETRv2-R50 | 43.0 | 53.4 | Transformers |
| 8 | RT-DETRv2-R34 | 31.5 | 49.9 | Transformers |
| 9 | RT-DETR-R50 | 43.0 | 53.1 | Transformers |
| 10 | RT-DETR-R18 | 20.2 | 46.5 | Transformers |

출처: [YOLO26](https://docs.ultralytics.com/models/yolo26/), [YOLO11](https://docs.ultralytics.com/models/yolo11/), [YOLOv10](https://docs.ultralytics.com/models/yolov10/), [RT-DETR 및 RT-DETRv2](https://github.com/lyuwenyu/RT-DETR/blob/main/README.md), [RTMDet](https://github.com/open-mmlab/mmdetection/blob/main/configs/rtmdet/README.md). YOLOv10의 파라미터 수는 공개 비교표의 fused 모델 수치이므로 체크포인트 내부 구조의 수와 다를 수 있습니다.

## 실행

저장소 루트의 PowerShell에서:

```powershell
.venv\Scripts\python.exe -m pip install -U ultralytics transformers
.venv\Scripts\python.exe scripts\evaluate_coco_candidates.py --source 'G:\내 드라이브\side_project\sample' --device cpu
```

10개를 한 번에 돌리면 CPU에서 오래 걸립니다. 우선 모델 두 개만 확인하려면:

```powershell
.venv\Scripts\python.exe scripts\evaluate_coco_candidates.py --source 'G:\내 드라이브\side_project\sample' --device cpu --models yolo26l yolo26m
```

각 모델의 가중치 다운로드와 이미지 한 장 추론만 확인하려면 같은 명령 끝에 `--smoke`를 붙이세요. 이 옵션은 전체 평가 결과를 쓰지 않습니다.

GPU가 있으면 `--device cuda:0`을 사용할 수 있습니다. 첫 실행에서 공식 가중치를 인터넷에서 내려받습니다. 각 모델의 가중치는 `models/temporary` 아래 임시 폴더에 저장했다가 해당 모델 평가가 끝나면 삭제합니다. 실행 도중 강제 종료하면 임시 폴더가 남을 수 있습니다. 모델 목록은 `--list-models`로 확인할 수 있습니다.

데이터는 `--source` 바로 아래 `laptop/`(또는 `computer/`), `book/`, 기타 비대상 폴더로 구성합니다. `laptop/`은 평가 라벨 `computer`로 변환합니다. 재귀적으로 이미지를 찾고 SHA-256이 같은 이미지는 한 번만 셉니다. 기존 평가와 비교하려면 고유 이미지가 `computer` 200장, `book` 71장, `other` 101장이어야 합니다. 수가 다르면 실행을 중단합니다. 다른 데이터로 별도 실험할 때만 `--allow-different-sample`을 사용하세요.

입력은 640, 최소 탐지 점수 0.01, YOLO NMS IoU 0.6, 그룹 수락 임계값 0.05입니다. `laptop`·`tv`는 `computer`, `book`은 `book`, 나머지 COCO 클래스는 `other`로 묶고 그룹별 최고 탐지 점수의 Top-1을 고릅니다. 모델마다 점수 분포가 다르므로 이 고정 임계값 평가 후에는 **별도 검증 데이터**로 임계값을 조정해야 합니다.

결과는 `outputs/coco_small_detectors/summary.csv`와 모델별 `predictions.csv`, `metrics.json`에 저장됩니다. FPR은 `other`를 대상으로 잘못 수락한 비율, FNR은 `computer`·`book`을 `other`로 거절한 비율입니다. 요약표에는 RTMDet-x의 기존 FPR 1/101·FNR 21/271보다 낮은지와 각 5% 목표를 충족하는지도 표시합니다. 지연은 모델 로딩과 이미지 파일 읽기를 제외하고 전처리·추론·후처리를 포함한 1회 실행의 평균이며, RTMDet 기존 측정값과 측정 방식 차이가 있을 수 있습니다. FPR·FNR과 함께 같은 환경에서 측정한 모델 간 지연을 비교하세요.
