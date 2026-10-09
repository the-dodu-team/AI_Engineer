# RTMDet-x vs 선택 탐지 모델 Top-k 비교

동일한 고유 이미지 372장에서 RTMDet-x와 선택한 탐지 모델 3종을 비교했습니다.

- RTMDet-x
- RT-DETRv2-R50
- RT-DETRv2-R34
- LW-DETR Large

선택 모델 3종은 같은 추론 점수에서 Top-1과 Top-2를 함께 계산했습니다. RTMDet-x는 기존 Top-1 결과를 비교 기준으로 사용했습니다. 이 실험은 추가 학습이나 fine-tuning이 아니라 COCO 사전학습 가중치의 추론 후처리 비교입니다.

## 결론

- **자동 단일 판정 1순위는 RT-DETRv2-R50 Top-1**입니다. 정확도 95.70%, FNR 5.54%, FPR 0.99%로 RTMDet-x보다 정확도와 FNR이 좋았습니다.
- **FPR 최소화 모델은 RT-DETRv2-R34 Top-1**입니다. FPR 0.00%, 정확도 94.09%로 RTMDet-x와 정확도가 같고 CPU 지연은 더 짧았습니다.
- Top-2는 선택 모델 3종의 전체 FNR을 모두 0%로 낮췄지만 FPR이 33.66~65.35%로 크게 증가했습니다.
- 따라서 Top-2는 자동 승인보다 후속 분류기 또는 사용자 확인에 전달하는 recall 우선 shortlist로 사용하는 편이 적절합니다.
- 현재 평가에서 FNR과 FPR을 모두 5% 이하로 만족한 구성은 없습니다.

![Top-1과 Top-2 핵심 지표](docs/reports/selected-topk2/topk-overview.png)

## Top-1 결과

전체 FNR은 `computer`·`book` 이미지를 `other`로 거절한 비율이고, 전체 FPR은 `other` 이미지에 대상 클래스를 수락한 비율입니다.

| 모델 | 정답/372 | 정확도 | FNR | FPR | CPU 지연¹ |
|---|---:|---:|---:|---:|---:|
| **RT-DETRv2-R50** | **356** | **95.70%** | **5.54%** | 0.99% | 853.1 ms |
| **RT-DETRv2-R34** | 350 | 94.09% | 7.75% | **0.00%** | **534.4 ms** |
| **RTMDet-x** | 350 | 94.09% | 7.75% | 0.99% | 1,346.7 ms |
| **LW-DETR Large** | 348 | 93.55% | 8.12% | 0.99% | 818.3 ms |

¹ 이미지 디스크 읽기와 모델 로딩을 제외한 CPU 이미지당 평균입니다. 프레임워크와 반복 횟수 차이가 있으므로 지연 시간은 참고값입니다.

![선택 모델 Top-1과 RTMDet-x 비교](docs/reports/selected-topk2/top1-vs-rtmdet.png)

## Top-1 vs Top-2

정답 포함률은 폴더 정답이 반환 후보 안에 존재하는 비율입니다. Top-2의 정답 포함률은 단일 분류 정확도가 아닙니다.

| 모델 | k | 정답 포함 | 정답 포함률 | 전체 FNR | 전체 FPR | 복수 후보율 |
|---|---:|---:|---:|---:|---:|---:|
| RT-DETRv2-R50 | 1 | 356/372 | 95.70% | 5.54% | 0.99% | 0.00% |
| RT-DETRv2-R50 | 2 | **371/372** | **99.73%** | **0.00%** | 53.47% | 87.37% |
| RT-DETRv2-R34 | 1 | 350/372 | 94.09% | 7.75% | 0.00% | 0.00% |
| RT-DETRv2-R34 | 2 | **370/372** | **99.46%** | **0.00%** | 65.35% | 90.59% |
| LW-DETR Large | 1 | 348/372 | 93.55% | 8.12% | 0.99% | 0.00% |
| LW-DETR Large | 2 | **368/372** | **98.92%** | **0.00%** | **33.66%** | 81.45% |

Top-2로 복구한 정답보다 새로 증가한 오수락이 더 많았습니다.

| 모델 | 추가 정답 포함 | 추가 오수락 |
|---|---:|---:|
| RT-DETRv2-R50 | +15 | +53 |
| RT-DETRv2-R34 | +20 | +66 |
| LW-DETR Large | +20 | +33 |

![Top-2 복구량과 오수락 비용](docs/reports/selected-topk2/recovery-vs-cost.png)

![FNR과 FPR 트레이드오프](docs/reports/selected-topk2/fnr-fpr-tradeoff.png)

## 클래스별 결과

Top-2에서 `computer`와 `book` recall은 크게 개선됐지만 precision과 FPR이 악화됐습니다. 특히 RT-DETRv2-R34 Top-2의 book precision은 51.82%, book FPR은 21.93%였습니다.

![클래스별 Precision·Recall·FNR·FPR](docs/reports/selected-topk2/class-metrics.png)

| 모델 | k | Computer Precision | Computer Recall | Book Precision | Book Recall |
|---|---:|---:|---:|---:|---:|
| RT-DETRv2-R50 | 1 | 100.00% | 97.50% | 98.39% | 85.92% |
| RT-DETRv2-R50 | 2 | 90.09% | 100.00% | 60.68% | 100.00% |
| RT-DETRv2-R34 | 1 | 100.00% | 95.50% | 98.31% | 81.69% |
| RT-DETRv2-R34 | 2 | 90.41% | 99.00% | 51.82% | 100.00% |
| LW-DETR Large | 1 | 100.00% | 92.00% | 96.97% | 90.14% |
| LW-DETR Large | 2 | 94.71% | 98.50% | **63.39%** | 100.00% |

## 평가 조건

- 고유 이미지 372장: computer 200, book 71, other 101
- 그룹 수락 임계값 0.05, 원시 탐지 하한 0.01
- 입력 크기 640 또는 체크포인트 기본 전처리
- CPU 10 threads, warm-up 5회, 이미지당 측정 1회
- 선택 모델 데이터 manifest SHA-256을 RTMDet-x 평가 데이터와 대조

모델 선정에 사용한 데이터와 같은 평가 세트이므로 최종 정책과 임계값은 별도 검증 데이터에서 확정해야 합니다. LW-DETR Large는 Objects365 사전학습 가중치의 상업 이용 조건도 별도로 확인해야 합니다.

## 실행과 상세 결과

- [실행 방법](docs/selected-topk2.md)
- [상세 Top-k 보고서](docs/reports/selected-topk2/README.md)
- [통합 수치 CSV](docs/reports/selected-topk2/comparison-with-rtmdet.csv)
- [클래스별 수치 CSV](docs/reports/selected-topk2/class-metrics.csv)
