# Speech Recognize (화자 식별) - 개발 표준

문서 버전: 1.0
대상 독자: 백엔드/AI 개발자, DevOps
관련 문서: [`./development-environment.md`](./development-environment.md)

---

## 목차

1. 개요
2. 개발환경
   2.1 개발환경 구성도
   2.2 개발절차
   2.3 개발자 PC 구성 내역
   2.4 IDE (Cursor / VSCode / PyCharm)
   2.5 소스 관리 (사내 Git + GitHub 미러)
   2.6 모델 / 패키지 / 직원 음성 DB 저장소
   2.7 IDE 설정 및 런타임 설치
       2.7.1 IDE 설정 (Cursor / VSCode)
       2.7.2 Python / uv 설치
       2.7.3 CUDA 12.1 / NVIDIA Container Toolkit
       2.7.4 시스템 의존성 (ffmpeg)
       2.7.5 Docker / Podman
       2.7.6 ModelScope ERes2Net 모델 배치
       2.7.7 직원 enrollment 음성 배치
3. 디렉토리 & 모듈 표준
4. 의존성 / 패키지 관리 표준
5. 코드 스타일 표준
6. API 표준
7. 모델 / 오디오 운영 규칙
8. Job 관리 표준
9. Docker / 배포 표준
10. 로깅 / 관측
11. 보안 / 개인정보
12. Git / 브랜치 / PR
13. 백엔드 연동 시 주의

---

## 1. 개요

본 문서는 화자 식별 서비스(`/home/pps-nipa/jenkins/dev/speech_recognize`)의 **개발 환경 / 모델 / 코드 / 배포** 표준을 정의합니다.
서비스는 **ERes2Net (3D-Speaker)** 모델로 사내 직원 음성 enrollment 와 Whisper STT 결과를 비교하여 **각 발화 구간의 화자**를 식별하는 **비동기 Job 기반 API** 입니다.

| 구분 | 기술 |
|------|------|
| 언어 | Python ≥ 3.9 (Docker 기본 3.10) |
| API | FastAPI + Uvicorn + BackgroundTasks |
| 모델 | ERes2Net (ModelScope `iic/speech_eres2net_base_sv_zh-cn_3dspeaker_16k`) |
| 추론 백엔드 | PyTorch 2.1 + CUDA 12.1, torchaudio |
| 패키지 매니저 | uv |
| 컨테이너 | Docker / Podman |
| CI | Jenkins (`/home/pps-nipa/jenkins/`) |

---

## 2. 개발환경

### 2.1 개발환경 구성도

```
┌──────────────────────────────────────────────────────────────────┐
│                          개발자 PC                                │
│   Cursor IDE  ──────────  Python 3.10 + uv (.venv)                │
│        │                        │                                 │
│        │ SSH/HTTPS              │ docker (GPU)                    │
└────────┼────────────────────────┼─────────────────────────────────┘
         │                        │
         ▼                        ▼
┌─────────────────────┐   ┌───────────────────────────────────────┐
│  사내 Git (Gitea)    │   │  ModelScope Hub / 사내 NAS 미러         │
│  narea/              │   │  iic/speech_eres2net_base_sv_*         │
│  speech_recognize.git│   └───────────────────────────────────────┘
└────────┬────────────┘                  │
         │                                ▼
         ▼                  ┌───────────────────────────────────────┐
┌────────────────────┐      │ src/resoursces/                        │
│ GitHub 미러         │      │  ├── models/iic/...                    │
│ GiJeongCho/         │      │  └── employee/<직원폴더>/<wav>          │
│ speech_recognize    │      └───────────────────────────────────────┘
└────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────────────────────────┐
│                Jenkins 서버 (Build / Deploy)                      │
│  dev/docker.sh dev up speech_recognize                            │
│        │                                                          │
│        ▼                                                          │
│   ┌──────────────────────────────────────────┐                    │
│   │ speech_recognize (FastAPI + BG Tasks)    │ ◄── /v1/recognize  │
│   │ ERes2Net pipeline pinned on GPU          │                    │
│   └──────────────────────────────────────────┘                    │
└──────────────────────────────────────────────────────────────────┘
```

### 2.2 개발절차

1. 개발자 PC에 IDE, Python 3.10, uv, Docker, NVIDIA 드라이버를 설치한다.
2. SSH 키 등록(사내 Git, GitHub).
3. `git clone ssh://git@git.biz.ppsystem.co.kr:10022/narea/speech_recognize.git`.
4. `uv sync` 로 의존성 설치.
5. `ffmpeg` 설치(`apt-get install ffmpeg`).
6. ERes2Net 모델을 `src/resoursces/models/iic/...` 에 배치 (또는 사내 NAS 마운트).
7. 직원 enrollment 음성을 `src/resoursces/employee/<직원ID>/<sample>.wav` 에 배치.
8. `uv run uvicorn src.api:app --host 0.0.0.0 --port 8016 --reload` 실행 → `/health` 확인.
9. PR → 사내 Git push → GitHub 미러 자동 반영.
10. Jenkins Job 트리거 → dev/stg/prd 배포 → 스모크 테스트.

### 2.3 개발자 PC 구성 내역

| 항목 | 최소 | 권장 | 비고 |
|------|------|------|------|
| OS | Ubuntu 20.04 LTS | Ubuntu 22.04 LTS | |
| CPU | 4 core | 8 core+ | |
| RAM | 16 GB | 32 GB | torchaudio 로딩 시 메모리 |
| Disk | 50 GB | 200 GB SSD | 모델 ≈ 1 GB, 직원 음성 별도 |
| GPU | 없음(CPU 가능) | RTX 3060 12GB+ | CUDA 12.1 호환 |
| Python | 3.10 | 3.10.x | |
| Docker / Podman | 24.x / 4.x | 최신 | `nvidia.com/gpu` 디바이스 지원 |
| ffmpeg | 4.x | 6.x | |

### 2.4 IDE (Cursor / VSCode / PyCharm)

- 권장: Cursor 또는 VSCode.
- 필수 확장:
  - **Python**, **Pylance**
  - **Ruff**
  - **Docker**
  - **REST Client** (`http://localhost:8016/v1/recognize` 호출 디버깅)
  - **Even Better TOML**

### 2.5 소스 관리 (사내 Git + GitHub 미러)

- 사내 Git: `ssh://git@git.biz.ppsystem.co.kr:10022/narea/speech_recognize.git`
- GitHub 미러: `https://github.com/GiJeongCho/speech_recognize.git`
- `origin` 에 fetch 1 + push 2. `git push origin <branch>` 한 번으로 동시 반영.

### 2.6 모델 / 패키지 / 직원 음성 DB 저장소

| 자원 | 저장소 | 비고 |
|------|--------|------|
| ERes2Net 모델 | ModelScope Hub / 사내 NAS 미러 | 오프라인 운영 시 NAS 미러 필수 |
| Python 패키지 | PyPI / 사내 Nexus | numpy < 2 (Docker), torch 2.1 cu121 |
| Docker 이미지 | 사내 Registry (`IMG_SPEECH_SR`) | dev/stg/prd 태그 분리 |
| 직원 enrollment 음성 | 사내 NAS / 별도 권한 디렉토리 | **민감 개인정보(생체)** — 별도 백업/접근통제 |

### 2.7 IDE 설정 및 런타임 설치

#### 2.7.1 IDE 설정 (Cursor / VSCode)

`.vscode/settings.json`:
```json
{
  "python.defaultInterpreterPath": ".venv/bin/python",
  "python.analysis.typeCheckingMode": "basic",
  "editor.formatOnSave": true,
  "[python]": {
    "editor.defaultFormatter": "charliermarsh.ruff",
    "editor.tabSize": 4
  },
  "files.exclude": {
    "**/src/resoursces/employee/**": true,
    "**/__pycache__": true
  }
}
```

`.vscode/launch.json`:
```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Speech Recognize (uvicorn)",
      "type": "debugpy",
      "request": "launch",
      "module": "uvicorn",
      "args": ["src.api:app", "--host", "0.0.0.0", "--port", "8016", "--reload"],
      "env": {
        "APP_PORT": "8016",
        "PYTHONPATH": "${workspaceFolder}",
        "EMPLOYEE_DB_PATH": "${workspaceFolder}/src/resoursces/employee"
      },
      "console": "integratedTerminal",
      "justMyCode": false
    }
  ]
}
```

#### 2.7.2 Python / uv 설치

```bash
# Conda 사용 시
conda create -n speech_recognize python=3.10 -y
conda activate speech_recognize

# uv
curl -LsSf https://astral.sh/uv/install.sh | sh

cd /home/pps-nipa/jenkins/dev/speech_recognize
uv sync
```

#### 2.7.3 CUDA 12.1 / NVIDIA Container Toolkit

```bash
# 드라이버 확인 (CUDA 12.1+ 호환 R535+)
nvidia-smi

# Container Toolkit (Docker)
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# Podman 사용 시 CDI 등록
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
```

#### 2.7.4 시스템 의존성 (ffmpeg)

```bash
sudo apt-get install -y ffmpeg
ffmpeg -version
```

#### 2.7.5 Docker / Podman

```bash
# Docker
curl -fsSL https://get.docker.com | sh

# 또는 Podman (README 권장)
sudo apt-get install -y podman
```

#### 2.7.6 ModelScope ERes2Net 모델 배치

```bash
# 옵션 A) modelscope CLI
pip install modelscope
python -c "
from modelscope import snapshot_download
snapshot_download('iic/speech_eres2net_base_sv_zh-cn_3dspeaker_16k',
  cache_dir='/home/pps-nipa/jenkins/dev/speech_recognize/src/resoursces/models')
"

# 옵션 B) 사내 NAS 미러
ln -s /mnt/nas/models/iic/speech_eres2net_base_sv_zh-cn_3dspeaker_16k \
       src/resoursces/models/iic/speech_eres2net_base_sv_zh-cn_3dspeaker_16k
```

#### 2.7.7 직원 enrollment 음성 배치

```
src/resoursces/employee/
├── EMP_0001/         # 사번/익명ID 권장 (실명 노출 금지)
│   ├── sample01.wav
│   └── sample02.wav
└── EMP_0002/
    └── sample01.wav
```
- WAV 16kHz mono 권장 (코드가 자동 변환).
- 화자당 최소 5초 이상, 노이즈 적은 발화 5~10개 권장.
- 이 디렉토리는 **이미지/Git에 포함 금지**. 볼륨 마운트로만.

---

## 3. 디렉토리 & 모듈 표준

| 레이어 | 위치 | 책임 |
|--------|------|------|
| API | `src/api.py` | FastAPI 앱 생성, lifespan, `/health`, 라우터 부착 |
| 라우터 | `src/v1/router.py` | `/v1/*` 엔드포인트, 입력 검증, BackgroundTasks |
| 엔진 | `src/v1/main.py` | `SpeakerEngine` (ERes2Net pipeline) |
| 유틸 | `src/v1/utils/` | Job 관리, Whisper JSON 파서 |
| 자원 | `src/resoursces/` | 모델 / 직원 enrollment 음성 |

> **금지**: 라우터에서 직접 `modelscope.pipeline` 호출하지 않는다. 항상 `SpeakerEngine` 을 통한다.

### 3.1 API 버저닝
- 현재: `/v1`. 라우터 = `APIRouter(prefix="/v1", tags=["speaker"])`.
- 신모델/포맷 변경 시 `src/v2/router.py` 신설 + `app.include_router(router_v2)`.

### 3.2 폴더명 규칙
- `resoursces` 폴더는 오타이지만 **외부 마운트/배포 스크립트가 의존** → 변경 금지. 신규 도입 시에만 `resources` 표기.

### 3.3 네이밍
- 함수/모듈: `snake_case`
- 클래스: `PascalCase` (`SpeakerEngine`, `JobInfo`)
- 상수: `UPPER_SNAKE`
- 엔드포인트: 동사형(`/recognize`, `/refine-json`)

---

## 4. 의존성 / 패키지 관리 표준

- 로컬: **`uv` + `pyproject.toml`**. `uv sync` 로 재현.
- Docker: `Dockerfile` 에 **버전 핀된 pip install** 목록 유지 (재현성 우선).
- 추가 시:
  1. `pyproject.toml` 갱신.
  2. `Dockerfile` pip 목록도 동일하게 핀.
  3. 둘이 불일치하면 PR 차단.
- 핵심 버전:
  - `torch>=2.1` (Docker: cu121), `torchaudio>=2.1`, `numpy<2` (Docker 빌드 시), `modelscope>=1.34`, `datasets<3.0`.

---

## 5. 코드 스타일 표준

- PEP8, 4-space, 100자.
- 함수/클래스 docstring 필수(`SpeakerEngine.identify_speaker` 예).
- 타입 힌트 의무 — 특히 라우터, 엔진 public API.
- `print` 금지 → `logger`.

### 5.1 로깅
- 모듈별 `logger = logging.getLogger(__name__)`.
- 작업 로그에 **`job_id`** 필수 포함.
- 예외 시 `logger.exception(...)`.

### 5.2 예외 처리
- 입력 검증 실패: `HTTPException(400, ...)`.
- 시스템/리소스 오류: `HTTPException(500, ...)`.
- 백그라운드 작업 예외는 **반드시 `JobManager.fail_job`** 으로 상태 갱신 후 종료.
- 업로드 임시파일은 `finally` 에서 정리.

---

## 6. API 표준

### 6.1 엔드포인트
| 메서드 | 경로 | 설명 |
|--------|------|------|
| GET | `/health` | 항상 `{"status":"healthy"}` |
| POST | `/v1/recognize` | Job 생성. 즉시 `{job_id, status}` 반환 |
| GET | `/v1/jobs/{job_id}` | Job 상태/결과 |
| POST | `/v1/refine-json` | Whisper JSON 후처리 |

> Job 패턴: **모든 장시간 추론은 BackgroundTasks 로 분리**. 동기 응답 변환 금지.

### 6.2 입력 검증
- Whisper JSON 파싱은 `utils/json_paser.extract_segments` 만 사용.
- 잘못된 포맷일 때 **루트 키 목록과 힌트를 응답 메시지에 포함** (현재 패턴 유지).
- `mic_output_json` 형식: `dict {results: [...]}` / `dict {result: {results: [...]}}` / 단순 list 모두 허용.

### 6.3 응답
- Job 결과는 `result.results` 배열에 `start, end, text, speaker, score` 포함.
- 임계값 `threshold` 미만의 매칭은 `unknown` / `null` 로 표기(엔진 책임).

---

## 7. 모델 / 오디오 운영 규칙

### 7.1 ERes2Net 파이프라인
- 항상 16k mono 로 정규화 (`SpeakerEngine.ensure_mono_16k`).
- 디바이스 결정은 `torch.cuda.is_available()` 만 신뢰.
- 모델은 lifespan 시점에 한 번만 로드하여 GPU에 상주 → 핫리로드 금지(전체 재시작).

### 7.2 직원 enrollment
- 화자당 최소 5초 이상, 노이즈 적은 음성.
- enrollment 변경 시 컨테이너 재시작 (현재는 매 요청마다 디렉토리 조회 → 재시작 불필요하지만, 캐시 도입 시 표준 개정).
- 폴더 = 라벨, **개인정보는 폴더명에 직접 노출하지 않는다(사번/익명ID 권장).**

### 7.3 임계값
- 기본 `threshold=0.2` (라우터) / `0.25` (README 권장). 운영 환경 튜닝 후 백엔드가 명시적 전달.
- 임계값/매칭 알고리즘 변경은 PR에 **품질 메트릭(EER, accuracy)** 첨부.

---

## 8. Job 관리 표준

- `utils/job.JobManager` 가 단일 진입점. 라우터/엔진 모두 이를 통해 상태 갱신.
- 상태 전이: `pending → processing → (completed | failed)`.
- 진행률: `update_progress(p)` (0.0 ~ 100.0).
- **메모리 기반 (단일 프로세스 가정).** 다중 워커/스케일 아웃 시 외부 저장소(Redis) 로 교체 필요 → 향후 표준 개정.

---

## 9. Docker / 배포 표준

- Base image: `pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime`. 변경 시 호환성 PR.
- `EXPOSE ${APP_PORT}` / `CMD ["sh","-c","python -m uvicorn ..."]` 유지.
- `.dockerignore`: `src/resoursces/models/`, `src/resoursces/employee/`, `.venv/`, `nohup.out`, `__pycache__/`.
- 모델/직원 음성은 **이미지에 포함 금지**. 볼륨 마운트.

### 9.1 Jenkins 연동
- `IMG_SPEECH_SR=<registry>/speech-recognize:<env>`.
- compose 서비스명: `speech_recognize`.
- 환경변수: `SPEAKER_MODEL_REL`, `RESOURCE_DIR_REL` (`docker.sh` 참조).
- 백엔드 URL: `SPEECH_SR_BASE=http://speech_recognize:${SPEECH_SERVICE_PORT}` (dev 6016 / stg 9016 / prd 8016).

### 9.2 헬스체크
```yaml
healthcheck:
  test: ["CMD", "curl", "-fsS", "http://localhost:${APP_PORT}/health"]
  interval: 30s
  timeout: 5s
  retries: 5
```

---

## 10. 로깅 / 관측

| 항목 | 표준 |
|------|------|
| 요청 로그 | `job_id`, 파일명, threshold |
| 진행률 로그 | 10% 단위 INFO, 나머지 DEBUG |
| 완료 로그 | `Job {id} completed in Ns` |
| 실패 로그 | `logger.exception` + `JobManager.fail_job` |
| `/health` | 200/`healthy` 보장 |

---

## 11. 보안 / 개인정보

- 직원 enrollment 음성은 **민감 개인정보(생체정보)** 로 취급.
  - 컨테이너 외부 노출 금지.
  - 백업/스냅샷에서도 마스킹/암호화.
  - 접근권한은 별도 RBAC.
- 업로드 음성/JSON 은 처리 직후 삭제(현재 `finally` 패턴 유지).
- 응답에 화자명을 그대로 노출하는 것이 정책에 맞는지 백엔드와 합의(필요 시 ID 매핑 후 노출).

---

## 12. Git / 브랜치 / PR

- 브랜치: `feat/speech-<topic>`, `fix/speech-<topic>`.
- 커밋: `[speech] <동사> <내용>`.
- 모델/음성 파일은 절대 커밋 금지(`*.wav`, `*.bin`, `*.onnx`).
- enrollment 추가/변경은 **별도 데이터 PR** 로 분리(소스 변경과 분리하여 추적성 확보).
- 사내 Git + GitHub 미러 동시 push.

---

## 13. 백엔드 연동 시 주의

- 본 서비스는 **STT 결과(JSON) 가 선행되어야 동작**. 호출 순서: `stt → speech_recognize`.
- 큰 음성(수십 분)은 처리 시간이 길다 → 백엔드는 **Job 폴링** 또는 webhook 패턴.
- 동시 호출은 GPU 자원에 따라 제한. 부하 테스트 후 rate-limit 적용.
- 결과 `speaker` 라벨은 enrollment 폴더명(사번/익명ID). 사용자 친화 이름 매핑은 백엔드 책임.
