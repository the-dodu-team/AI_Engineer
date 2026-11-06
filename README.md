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

## Vercel 배포 (컨테이너)

PC가 꺼져 있어도 동작하도록 Vercel Container Runtime(베타)으로 배포합니다. 루트 `Dockerfile`이 이미지를 만들고 Vercel이 실행합니다. 코드는 로컬 실행과 같습니다.

- **주소**: `https://dinov2-api-one.vercel.app` (Vercel 프로젝트 `dinov2-api`, 팀 GJ)
- **자동 배포**: GitHub `main`에 푸시하면 운영 배포가 새로 만들어집니다(빌드 약 2분). 다른 브랜치는 미리보기 배포가 됩니다.
- **환경변수**: Vercel 프로젝트의 `DINOV2_API_KEY`(Sensitive). 백엔드가 `X-API-Key` 헤더로 보내는 값과 같아야 합니다. 키를 바꾸려면 값을 수정하고 다시 배포하세요.
- **모델**: DINOv2-base 가중치를 이미지에 넣어 두어서 시작할 때 다운로드하지 않습니다(`HF_HUB_OFFLINE=1`).

**동작과 제한** (Hobby 플랜 기준, 실제 측정값)

| 항목 | 내용 |
|---|---|
| 요청 크기 | **본문 4.5 MB 이하**. 넘으면 `413 FUNCTION_PAYLOAD_TOO_LARGE`. 배치는 이미지 합계가 4.5 MB 미만이어야 함 |
| 콜드 스타트 | 5분 이상 요청이 없으면 서버가 쉬고, 다음 첫 요청이 **약 8초** 걸림 |
| 일반 응답 | 이미지 1장 약 1.3초(서버 처리 약 0.5초 + 한국에서의 네트워크), 20장 배치 약 10초 |
| 메모리 | 2 GB / 1 vCPU. 20장 배치까지 정상 처리 확인 |
| 지역 | `iad1`(미국 동부) |
| 최대 실행 시간 | 300초 |

Hobby 플랜은 비상업·개인 용도 약관이 있습니다. 팀 서비스로 계속 쓸 때는 플랜을 확인하세요.

## 로컬 실행

처음 실행할 때 Hugging Face에서 `facebook/dinov2-base`(약 350 MB)를 내려받습니다.

**설치**

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

**서버 시작**

```bash
export PYTHONPATH=. DINOV2_ARTIFACT=artifacts/dinov2_base_prototypes.npz DINOV2_API_KEY=<임의의 키> DINOV2_DEVICE=cpu
python -m uvicorn dinov2_api.app:app --host 0.0.0.0 --port 8000 --workers 1
```

PowerShell에서는 아래처럼 설정합니다.

```powershell
$env:PYTHONPATH = "."
$env:DINOV2_ARTIFACT = "artifacts\dinov2_base_prototypes.npz"
$env:DINOV2_API_KEY = "<임의의 키>"
$env:DINOV2_DEVICE = "cpu"
python -m uvicorn dinov2_api.app:app --host 0.0.0.0 --port 8000 --workers 1
```

`http://127.0.0.1:8000/health`와 `http://127.0.0.1:8000/docs`에서 확인합니다.

**인터넷에서 호출하게 하려면 (ngrok)**

다른 터미널에서 ngrok으로 열어 줍니다. ngrok을 설치하고 authtoken을 등록한 상태여야 합니다.

```bash
ngrok http 8000 --url=<고정 도메인>
```

고정 도메인이 없으면 `--url` 없이 실행하면 임시 주소가 만들어집니다. 무료 주소는 요청 헤더에 `ngrok-skip-browser-warning: 1`을 넣으면 안내 화면 없이 응답합니다. 외부에 열 때는 `DINOV2_API_KEY`를 반드시 설정하세요.

**점검**

```bash
DINOV2_API_KEY=<서버와 같은 키> python scripts/smoke_test.py --base-url http://127.0.0.1:8000
```

상태 확인, 인증 차단, 단일·배치 예측, 잘못된 이미지 차단을 확인하며 키는 출력하지 않습니다.

## 구성

| 경로 | 내용 |
|---|---|
| `dinov2_api/` | FastAPI 앱과 예측 코드 |
| `artifacts/dinov2_base_prototypes.npz` | prototype과 판정 기준(평균 벡터 값) |
| `scripts/smoke_test.py` | 서버 점검 스크립트 |

`artifacts/dinov2_base_prototypes.npz`는 COCO·Open Images 이미지에서 계산한 평균 벡터이며 원본 이미지는 포함하지 않습니다. 학습 이미지에 COCO의 NonCommercial 라이선스가 섞여 있어, 상업적으로 쓰기 전에 데이터 라이선스를 확인해야 합니다.
