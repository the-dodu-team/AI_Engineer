# Embedding 기반 laptop / book 인증 실험

목적: 사용자가 찍은 사진이 **laptop**, **book**, **other** 중 무엇인지 판정한다. **학습과 임계값 결정에는 laptop과 book만 쓰고**, other는 평가(test)에만 쓴다. 처음 보는 `other`도 거절해야 하므로 사전학습 embedding과 "학습 이미지에서 얼마나 먼가"를 보는 방법으로 판정한다.

| 단계 | backbone | head | 코드 |
|---|---|---|---|
| 1 | 고정 | 학습 없음: prototype, kNN, Mahalanobis / 학습: logistic regression, 작은 MLP | `embed_verify.stage1_frozen` |
| 2 | 마지막 N개 블록 fine-tuning | cosine classifier | `embed_verify.stage2_finetune` |

후보 모델 4개: `dinov2-small`, `dinov2-base`, `clip-vit-b16`, `siglip-base`.

## 데이터와 split

데이터는 **`sampleV2` 폴더 하나**만 씁니다. 이 폴더 안에서 이미지를 train / val / test로 나눕니다. 이전 탐지 모델 평가용 372장 같은 별도 평가셋은 쓰지 않습니다(`sampleV2`와 이미지가 겹칠 수 있어 평가가 부풀려질 수 있기 때문입니다).

폴더 구조는 다음과 같습니다. 폴더 이름이 곧 라벨입니다.

```text
sampleV2/
  laptop/   -> laptop
  book/     -> book
  bed/ table/ bicycle/ guitar/ ...  -> 모두 other (폴더 이름은 other 종류)
```

split 규칙은 다음과 같습니다.

- 비율은 기본 train 0.6 / val 0.2 / test 0.2입니다. `--val-ratio`, `--test-ratio`로 바꿀 수 있습니다.
- **train과 val에는 laptop·book만 들어갑니다.** other는 전부 test에만 쓰므로(train 0장, val 0장) other의 개수가 적어도 평가에 모두 쓸 수 있습니다. other로 임계값도 맞추고 싶으면 `--other-in-val`, 학습에도 쓰고 싶으면 `--other-in-train`을 추가하세요.
- laptop·book은 `manifest.csv`의 `source`(COCO·Open Images)별로, other는 폴더(종류)별로 나눕니다. 그래서 모든 split에 두 출처와 모든 other 종류가 들어갑니다.
- 파일 내용이 같은 이미지(SHA-256)는 한 번만 남겨서 train과 test에 동시에 들어가지 않게 합니다. 크기만 다른 비슷한 이미지는 걸러내지 못합니다.
- split은 `--seeds`마다 다르게 나뉩니다(기본 0 1 2). 같은 seed는 Stage 1과 Stage 2에서 같은 split이라 서로 비교할 수 있습니다.
- 임계값은 val의 laptop·book 점수로만 정하고, 리포트의 결과는 test 기준입니다. 어떤 이미지가 어디에 들어갔는지는 `splits.csv`에 남습니다.

| 인자 | 내용 |
|---|---|
| `--train-dir` | 위 구조의 폴더 (`sampleV2`). `--other-dir`로 other 폴더를 따로 추가할 수도 있습니다. |
| `--other-dir` | (선택) 추가 other 폴더. 하위 폴더 이름이 other 종류가 되고, `book`·`laptop` 폴더는 자동으로 건너뜁니다. |
| `--val-ratio`, `--test-ratio` | validation(임계값·모델 선택용)과 test(결과 보고용)의 비율. 기본값은 각각 0.2입니다. |
| `--other-in-val` | (선택) other를 val(임계값 튜닝)과 test에 절반씩 나눕니다. 기본은 사용하지 않습니다. |
| `--other-in-train` | (선택) other도 학습에 사용합니다(`--other-in-val` 포함). 기본은 사용하지 않습니다. |
| `--unseen-other` | (선택) `--other-in-val`이나 `--other-in-train`을 켰을 때, 일부 other 종류를 test에만 쓰게 합니다. 기본 설정에서는 모든 other가 이미 test 전용이라 필요 없습니다. |

other 데이터를 만들 때 주의합니다.

- 책이나 노트북이 찍힌 사진을 other에 넣으면 라벨 오류가 되어 FPR이 부풀려집니다.
- 학습용 laptop·book이 crop이면 other도 비슷한 형태(crop 또는 같은 종류의 사진)로 맞추세요. 형태가 다르면 모델이 "crop인지 아닌지"를 배울 수 있습니다.

## 실행 (VS Code 터미널, PowerShell)

```powershell
cd C:\Users\jgi01\Desktop\side_project
git switch experiment/embedding-verification
.\.venv\Scripts\python.exe -m pip install -r requirements-embed.txt
```

```powershell
$TRAIN = "G:\내 드라이브\side_project\sampleV2"
```

### 0. 환경 확인 (합성 이미지, 약 1분)

```powershell
.\.venv\Scripts\python.exe -m embed_verify.make_synthetic outputs\embed_verify\_synthetic
.\.venv\Scripts\python.exe -m embed_verify.stage1_frozen --train-dir outputs\embed_verify\_synthetic\train --other-dir outputs\embed_verify\_synthetic\other --models dinov2-small --seeds 0 --run-name smoke
```

합성 이미지라 점수는 의미가 없고, 끝까지 돌아가는지만 확인합니다.

### 1. Stage 1 (4개 모델, seed 3개)

```powershell
.\.venv\Scripts\python.exe -m embed_verify.stage1_frozen --train-dir $TRAIN --seeds 0 1 2 --run-name stage1-pos
```

처음 실행할 때 모델 가중치를 내려받습니다(약 2 GB).

### 2. Stage 2 (Stage 1에서 좋았던 모델만)

```powershell
.\.venv\Scripts\python.exe -m embed_verify.stage2_finetune --train-dir $TRAIN --models dinov2-small --seeds 0 1 2 --unfreeze-blocks 2 --epochs 8 --run-name stage2
```

CPU에서는 모델·이미지 수에 따라 오래 걸립니다. GPU가 있으면 `--device cuda:0`을 추가하세요. `--unfreeze-blocks`를 늘리면 더 많이 학습합니다.

### 3. 결과 합치기 (선택)

```powershell
.\.venv\Scripts\python.exe -m embed_verify.report --runs outputs\embed_verify\stage1-pos outputs\embed_verify\stage2 --out outputs\embed_verify\REPORT_all.md
```

## 출력물

`outputs\embed_verify\<run-name>\`에 생깁니다. **Claude에게 정리를 요청할 때는 `REPORT.md`(합친 경우 `REPORT_all.md`)를 주면 됩니다.**

| 파일 | 내용 |
|---|---|
| `REPORT.md` | 실험 설정, 모델별 FNR·FPR·AUROC, 고정 FPR 지점, 클래스별 recall, other 종류별 FPR, 속도·크기, 오류 사례 |
| `metrics.csv` | 모델·방법·seed·split·임계값 지점별 모든 지표와 confusion matrix |
| `summary.csv` | seed 평균과 표준편차 |
| `predictions.csv` | 이미지별 점수와 예측 |
| `splits.csv`, `data_manifest.csv` | 어떤 이미지가 어느 split에 들어갔는지, SHA-256 |
| `run_config.json` | 인자, 데이터 개수, 중복 제거 결과, git 커밋, 라이브러리 버전 |
| `latency.csv`, `history.csv` | 이미지 1장당 지연(Stage 1), epoch별 학습 기록(Stage 2) |

## 지표 정의

- **FNR**: laptop·book을 other로 거절한 비율
- **swap**: laptop을 book으로(또는 반대로) 잘못 맞힌 비율
- **FPR**: other를 laptop·book으로 받아들인 비율
- 모든 방법은 점수가 가장 높은 target 클래스를 고르고, 점수가 임계값 이상일 때만 받아들입니다. 임계값은 **검증 데이터로만** 정합니다.
  - `q99`, `q95`: val의 laptop·book 이미지 중 99%, 95%를 받아들이는 점수. other가 필요 없어서 기본 설정에서 쓰는 임계값입니다.
  - `oracle`: 모델마다 test에서 (FNR+FPR)/2가 가장 낮은 임계값 **하나**. 임계값 선택 때문에 생기는 차이를 없애고 모델끼리 비교하기 위한 값입니다. test에서 직접 고르므로 **낙관적인 상한**이고 실제 서비스에서는 쓸 수 없습니다. 리포트 2번 표가 이 값이고, 같은 방법의 `q99`·`q95` 결과를 옆에 함께 보여 줍니다.
  - `macro`, `fpr5`, `fpr1`: `--other-in-val`을 켰을 때만 나옵니다. val macro recall 최대, val FPR 5%·1% 이하.
- 같은 seed는 Stage 1과 Stage 2에서 같은 split을 만들어 서로 비교할 수 있습니다.

## 참고

- 이전 RT-DETRv2-R50 결과(FNR 5.5%, FPR 1.0%)는 다른 평가 데이터에서 나온 값이라 참고용으로만 리포트에 표시됩니다.
- 학습 이미지의 라이선스(COCO의 일부는 NonCommercial)는 `manifest.csv`에 기록돼 있으니 상업 이용 전에 확인해야 합니다.
- 중복 제거는 SHA-256이 같은 파일만 걸러내며, 크기만 다른 비슷한 이미지는 걸러내지 못합니다.
- 테스트: `.\.venv\Scripts\python.exe tests\test_embed_verify.py`
