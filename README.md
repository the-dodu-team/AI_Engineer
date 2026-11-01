# DINOv2-base laptop · book 분류 API

사진을 `laptop`, `book`, `other` 중 하나로 분류하는 FastAPI 서버입니다. DINOv2-base 모델과 prototype(laptop·book 이미지로 만든 기준)을 사용하며, 내부 판정 기준은 응답에 노출하지 않습니다.

**현재 버전: `v0.0.0`** (배포 전)

## API

| 엔드포인트 | 설명 |
|---|---|
| `GET /health` | 상태 확인 |
| `POST /predict` | 이미지 1장 (multipart 필드 `file`) |
| `POST /predict/batch` | 이미지 1~20장 (multipart 필드 `files` 반복) |
| `GET /docs` | Swagger UI |

- 인증: `X-API-Key` 헤더. 서버의 `DINOV2_API_KEY` 환경변수와 같아야 합니다(설정하지 않으면 인증이 꺼지므로 반드시 설정하세요).
- 이미지는 1장당 최대 10 MiB이고 서버에 저장하지 않습니다.
- 응답 예시: `{"model": "dinov2-base-prototype", "processing_ms": 123.4, "result": {"filename": "a.jpg", "prediction": "laptop"}}`

## GitHub Actions로 실행 (uvicorn + ngrok)

`.github/workflows/serve-dinov2-api.yml`이 러너에서 uvicorn으로 서버를 띄우고 ngrok으로 공개합니다. 수동 실행만 지원하며, 푸시나 PR로는 시작되지 않습니다.

**1. 처음 한 번 설정** (저장소 Settings > Secrets and variables > Actions)

| 종류 | 이름 | 값 |
|---|---|---|
| Secret | `DINOV2_API_KEY` | API 키(충분히 긴 임의 문자열) |
| Secret | `NGROK_AUTHTOKEN` | ngrok 대시보드의 authtoken |
| Variable (선택) | `NGROK_DOMAIN` | ngrok 고정 도메인(예: `xxx.ngrok-free.dev`). 비우면 실행할 때마다 주소가 바뀝니다 |

```bash
gh secret set DINOV2_API_KEY
gh secret set NGROK_AUTHTOKEN
gh variable set NGROK_DOMAIN --body "xxx.ngrok-free.dev"
```

`gh secret set`은 값을 물어보므로 터미널에 직접 붙여 넣으세요. 키를 명령어나 채팅에 적지 마세요.

**2. 실행**

```bash
gh workflow run "Serve DINOv2 API" -f duration_minutes=120
```

Actions 탭에서 실행해도 됩니다. 시작 후 의존성 설치, 모델 다운로드, 점검을 거쳐 보통 3~6분 뒤에 열립니다. 실행 요약(Summary)과 로그에 Base URL이 표시됩니다.

**3. 확인**

```bash
curl https://<Base URL>/health
DINOV2_API_KEY=... python scripts/smoke_test.py --base-url https://<Base URL>
```

ngrok 무료 주소는 요청 헤더에 `ngrok-skip-browser-warning: 1`을 넣으면 안내 화면 없이 응답합니다.

**제한 사항**
- 한 번에 최대 330분(5.5시간)까지 실행되고, 그 뒤에는 자동으로 종료됩니다. 계속 쓰려면 다시 실행해야 합니다.
- 동시에 하나만 실행됩니다(같은 ngrok 도메인을 두 번 쓸 수 없습니다).
- GitHub 호스트 러너는 CPU만 사용합니다. 이미지 1장당 약 0.2~0.3초가 걸립니다.
- 이 저장소는 공개 저장소라 Actions 로그와 Base URL이 외부에 보입니다. API 키는 로그에 마스킹되지만 URL은 노출되므로 반드시 `DINOV2_API_KEY`를 설정해야 합니다.
- 테스트와 UT 용도의 임시 서버입니다. 상시 운영에는 별도 호스팅을 사용하세요.

## 로컬 실행

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
export PYTHONPATH=. DINOV2_ARTIFACT=artifacts/dinov2_base_prototypes.npz DINOV2_API_KEY=<임의의 키> DINOV2_DEVICE=cpu
python -m uvicorn dinov2_api.app:app --host 127.0.0.1 --port 8000 --workers 1
```

PowerShell에서는 `$env:PYTHONPATH = "."`처럼 환경변수를 설정합니다. 처음 실행할 때 Hugging Face에서 `facebook/dinov2-base`(약 350 MB)를 내려받습니다.

## 구성

| 경로 | 내용 |
|---|---|
| `dinov2_api/` | FastAPI 앱과 예측 코드 |
| `artifacts/dinov2_base_prototypes.npz` | prototype과 판정 기준(평균 벡터 값) |
| `scripts/smoke_test.py` | 서버 점검 스크립트(키는 출력하지 않음) |
| `.github/workflows/serve-dinov2-api.yml` | 서버 실행 워크플로 |

`artifacts/dinov2_base_prototypes.npz`는 COCO·Open Images 이미지에서 계산한 평균 벡터이며 원본 이미지는 포함하지 않습니다. 학습 이미지에 COCO의 NonCommercial 라이선스가 섞여 있어, 상업적으로 쓰기 전에 데이터 라이선스를 확인해야 합니다.
