# Embedding 기반 laptop · book 인증 실험

브랜치: `experiment/embedding-verification`. 사진이 **laptop**, **book**, **other** 중 무엇인지 판정하는 모델을 비교한 실험입니다.
사전학습 임베딩 모델 4종과 기존 RT-DETRv2-R50(zero-shot)을 같은 이미지, 같은 split으로 비교했습니다. 임베딩 모델은 validation의 laptop·book 99%를 통과시키는 임계값을 쓰고, R50은 **기존 규칙 그대로**(사람·의자 등 다른 물체 검출 점수도 함께 사용, 임계값 0.05) 적용했습니다.
한 장 요약 PDF는 `outputs/embed_verify/summary.pdf`(git 제외)입니다. 실행 방법은 [docs/embedding-verification.md](../docs/embedding-verification.md)에 있습니다.

## 결론

- **가장 잘 나온 모델은 DINOv2-base (prototype)** 입니다. 정확도 99.0%, 놓침(FNR) 0.7%, 오인식(FPR) 0.0%입니다. 기존 규칙의 R50은 89.8%, 11.5%, 1.0%입니다.
- 후보 4종 중 **DINOv2 계열이 가장 좋았고**, CLIP·SigLIP은 laptop·book 이미지만 쓰는 이 방식에서는 other 거절이 약했습니다.
- 이미지 1장당 CPU 225 ms로 R50(941 ms)보다 약 4배 빠릅니다.
- **제안: UT에 DINOv2-base + prototype 적용.** UT 전에 휴대폰으로 촬영한 test 데이터를 클래스별 20장씩 구축해 확인하고, Backend FastAPI 호출 검증을 이 모델로 수행합니다.
- 기존 R50 보고 수치(FNR 5.5%, FPR 1.0%)와 이번 R50 수치 중 오인식은 같고, 놓침은 laptop·book 이미지가 달라서 다릅니다(아래 4번).

## 평가 조건

| 항목 | 내용 |
|---|---|
| 데이터 | sampleV2: laptop 1,437장, book 918장(COCO·Open Images에서 잘라낸 사진), other 101장(bed·bicycle·guitar·table) · [Drive 폴더](https://drive.google.com/drive/folders/1Cy3_bLk43XQuGOcoZis2s84aIqvqfdJz?usp=drive_link) |
| 기준 만들기 | laptop·book 이미지만 사용(train 60%). other는 학습·임계값 결정에 쓰지 않음. 모델 가중치 fine-tuning은 하지 않음 |
| 임계값 결정 | validation(20%)의 laptop·book 점수만 사용 |
| 평가(test) | seed마다 laptop 288장 · book 183장 · other 101장, 서로 다른 split(seed) 3개의 평균 |
| 지표 | FNR(놓침): laptop·book을 other로 판정한 비율, FPR(오인식): other를 laptop·book으로 판정한 비율 |
| 임계값 규칙 | **99%·95% 임계값**: validation laptop·book의 99%·95%를 통과시키는 점수(실제 적용 가능). **최적 임계값**: test에서 직접 고른 가장 좋은 값(모델 비교용 상한, 실제 적용 불가) |
| 측정 환경 | CPU 10 threads, 이미지 1장씩 처리(읽기·전처리·추론 포함) |

## 1. 모델별 비교

각 모델에서 가장 좋았던 방법입니다. 전체 방법 조합 결과는 실행 후 생기는 `REPORT_all.md`에 있습니다.

| 모델 | 가장 좋은 방법 | 최적 임계값 FNR / FPR | AUROC | 99% 임계값 FNR / FPR | 속도(1장) |
|---|---|---|---|---|---|
| DINOv2-base | prototype | 0.3% / 0.0% | 1.000 | 0.7% / 0.0% | 225 ms |
| DINOv2-small | kNN | 1.1% / 1.0% | 0.999 | 0.6% / 3.0% | 109 ms |
| SigLIP-base | logistic regression | 5.7% / 0.7% | 0.984 | 2.3% / 29.0% | 248 ms |
| CLIP ViT-B/16 | logistic regression | 5.3% / 1.7% | 0.987 | 2.0% / 19.5% | 216 ms |

- DINOv2가 가장 안정적입니다. DINOv2-base는 단순한 prototype(laptop·book 평균 벡터와의 유사도)만으로 AUROC 1.000에 이르렀습니다.
- CLIP·SigLIP은 가장 좋은 방법(logistic regression)도 놓침이 5% 이상이고, 학습 없이 거리만 보는 방법은 other를 거의 거절하지 못했습니다.
- other를 학습에 쓰지 않으면 일반 분류기(logistic regression·MLP)는 처음 보는 other에 점수를 높게 주는 경향이 있어 99% 임계값에서 오인식이 커졌습니다. 거리를 보는 방법(prototype·kNN)이 더 안정적이었습니다.
- other가 4종 101장이라 FPR은 1장 차이로도 1%포인트씩 움직입니다. FPR 0.0%는 '101장 중 0장'이며 95% 신뢰 상한은 약 3%입니다.

## 2. 가장 잘 나온 모델과 기존 RT-DETRv2-R50 비교

R50은 COCO 사전학습 모델을 학습 없이 그대로 쓴 결과입니다. **기존 규칙**은 laptop(+tv)·book 점수가 사람·의자 같은 다른 물체 검출 점수보다 높고 0.05 이상일 때만 laptop·book으로 판정하는 이전 방식입니다. 참고로 다른 물체 점수를 쓰지 않고 laptop·book 점수만으로 임베딩 모델과 같은 방식으로 임계값을 정한 결과도 함께 적었습니다.

| 모델 (규칙) | 정확도 | 놓침 FNR / 오인식 FPR | 최적 임계값 FNR / FPR | 속도(1장) | 크기(M) |
|---|---|---|---|---|---|
| DINOv2-base (prototype, 99% 임계값) | 99.0% | 0.7% / 0.0% | 0.3% / 0.0% | 225 ms | 87 |
| DINOv2-small (kNN, 99% 임계값) | 98.2% | 0.6% / 3.0% | 1.1% / 1.0% | 109 ms | 22 |
| **RT-DETRv2-R50 (기존 규칙, 임계값 0.05)** | 89.8% | 11.5% / 1.0% | 11.7% / 0.3% | 941 ms | 43 |
| RT-DETRv2-R50 (laptop·book 점수만, 99% 임계값, 참고) | 94.3% | 1.2% / 14.5% | 6.8% / 5.0% | 941 ms | 43 |

- DINOv2-base는 기존 규칙의 R50보다 정확도, 놓침, 오인식이 모두 좋습니다. 특히 놓침이 0.7% 대 11.5%로 차이가 큽니다. 최적 임계값끼리 비교해도 0.3% / 0.0% 대 11.7% / 0.3%입니다.
- 기존 규칙은 laptop·book 사진에서도 다른 물체(사람·키보드 등)가 더 강하게 검출되면 other로 판정해서 놓침이 커집니다.
- laptop·book 점수만 쓰고 95% 임계값을 쓰면 R50의 놓침 5.1%, 오인식 9.6%입니다. DINOv2-base의 95% 임계값은 4.3% / 0.0%입니다.

## 3. 학습 모델을 쓸지, zero-shot을 쓸지

| 기준 | 임베딩 방식 (DINOv2-base) | zero-shot (RT-DETRv2-R50, 기존 규칙) |
|---|---|---|
| 정확도 | 99.0% | 89.8% |
| 놓침 · 오인식 | 0.7% · 0.0% (99% 임계값) | 11.5% · 1.0% |
| 속도(CPU 1장) | 225 ms | 941 ms |
| 준비물 | laptop·book 이미지로 기준 벡터를 한 번 만들어 둠(이미 1,400장 이상 확보) | 없음 |
| other 거절 방식 | 'laptop·book 사진과 얼마나 다른가'를 거리로 판단 | COCO의 80개 물체 인식 결과를 활용(목록에 있는 물체에 의존) |
| 남은 위험 | 실제 사용자 사진 미검증(UT 전 휴대폰 데이터로 확인 예정) | 놓침이 높고 속도가 느림 |

**권고**: DINOv2-base + prototype(99% 임계값)으로 전환합니다. 속도가 더 중요하면 DINOv2-small(kNN, 109 ms)이 대안이며, R50은 비교 기준과 대체 수단으로 유지합니다.

## 4. 이전 R50 수치와 이번 수치가 다른 이유

기존 보고서의 R50은 놓침 5.5%, 오인식 1.0%였고, 이번에 같은 규칙으로 평가한 결과는 놓침 11.5%, 오인식 1.0%입니다.

| 항목 | 기존 R50 보고 | 이번 비교 |
|---|---|---|
| 판정 규칙 · 임계값 | other 물체 검출 점수를 포함한 세 그룹 경쟁, 고정 0.05 | **같음** |
| other 이미지 | 101장 | **같은 101장**(이미지 해시로 확인) → 오인식 1.0%로 일치 |
| laptop·book 이미지 | 372장 평가셋의 computer 200장, book 71장 | sampleV2에서 seed마다 laptop 288장, book 183장. **기존 평가 이미지와 겹치는 것이 0장** → 놓침이 다름 |
| 반복 | 1회 | 서로 다른 split 3개의 평균 |

- 오인식은 규칙과 other 이미지가 같아서 기존 수치(1.0%)가 그대로 재현됩니다.
- 놓침이 다른 것은 laptop·book 이미지가 달라졌기 때문입니다. 이번 이미지는 COCO·Open Images의 큰 사진에서 잘라낸 사진이라 기존 평가 이미지와 난이도가 다릅니다.
- 이전에 같은 R50을 laptop·book 점수만으로 평가했을 때(오인식 53.5%, 임계값 0.05)와 달랐던 것은 다른 물체 점수를 쓰지 않았기 때문입니다. 위 2번 표의 마지막 행처럼 임계값을 laptop·book 점수로 정하면 오인식은 14.5%(99% 임계값)입니다.

**이번 비교를 쓰는 이유**: 모든 모델을 같은 이미지, 같은 split으로 평가했으므로 **어떤 모델이 더 좋은지(순위와 차이)**를 비교하는 용도로 유효합니다. 절대 성능은 아래 한계를 해결한 뒤 다시 확인해야 합니다.

## 한계와 다음 단계

- other가 4종 101장뿐이라 FPR 0.0%의 95% 신뢰 상한은 약 3%입니다.
- laptop·book 이미지가 COCO·Open Images에서 잘라낸 사진이라, 사용자가 휴대폰으로 찍는 전체 사진과 다릅니다. 실제 사진 테스트셋이 필요합니다.
- fine-tuning(`stage2_finetune`)은 구현했지만 이번 비교에는 포함하지 않았습니다. 위 확인 후에도 부족하면 진행합니다.
- 학습 이미지에 COCO의 NonCommercial 라이선스가 섞여 있어 상업 이용 전에 확인이 필요합니다.

## 재현 방법

임베딩 모델과 R50 zero-shot을 같은 split으로 실행하고 하나의 리포트로 합칩니다. 자세한 설명과 옵션은 [docs/embedding-verification.md](../docs/embedding-verification.md)를 보세요.

```powershell
$TRAIN = "G:\내 드라이브\side_project\sampleV2"
python -m embed_verify.stage1_frozen --train-dir $TRAIN --seeds 0 1 2 --run-name stage1-pos
python -m embed_verify.baseline_rtdetr --train-dir $TRAIN --seeds 0 1 2 --compare-with outputs\embed_verify\stage1-pos --run-name baseline-rtdetrv2-full
python -m embed_verify.report --runs outputs\embed_verify\stage1-pos outputs\embed_verify\baseline-rtdetrv2-full --out outputs\embed_verify\REPORT_all.md
```

## 코드 구성

| 파일 | 역할 |
|---|---|
| `data.py` | 이미지 읽기, 중복 제거, train/val/test split |
| `backbones.py` | DINOv2·CLIP·SigLIP 임베딩 추출, 속도 측정 |
| `heads.py` | prototype·kNN·Mahalanobis·logistic regression·MLP |
| `evalcore.py` | 임계값 선택, 지표 계산, 결과 기록 |
| `stage1_frozen.py` | 1단계: backbone 고정 + head |
| `stage2_finetune.py` | 2단계: 마지막 블록 fine-tuning |
| `baseline_rtdetr.py` | RT-DETRv2-R50 zero-shot 기준선 |
| `report.py` | `REPORT.md` 생성 |
