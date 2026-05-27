# Speech Recognize (화자 식별) - 개발 환경 가이드

> 백엔드 개발자 대상 문서.
> 본 서비스는 **사내 직원 음성 DB** 와 **Whisper STT 결과**를 비교하여 **각 발화 구간이 누구의 목소리인지** 식별하는 **AI 백엔드**입니다.
> STT(음성 → 텍스트)는 별도 `stt` 서비스, 본 서비스는 **화자 인식(Speaker Identification)** 만 담당합니다.

---

## 1. 서비스 개요

| 항목 | 내용 |
|------|------|
| 서비스 이름 | Speaker Recognition API |
| 모델 | **ERes2Net** (3D-Speaker) — `iic/speech_eres2net_base_sv_zh-cn_3dspeaker_16k` (ModelScope) |
| Pipeline | `modelscope.pipelines.pipeline(task="speaker-verification", ...)` |
| 입력 | (1) 분석 음성 파일(WAV/MP3/M4A) (2) Whisper STT JSON(`chunks`/`segments`) (3) [선택] mic_speech_recognize JSON |
| 출력 | 각 발화 구간(start/end/text)에 `speaker` 가 매핑된 결과 |
| 처리 방식 | **비동기 Job 패턴** (`POST /v1/recognize` → `job_id` → `GET /v1/jobs/{job_id}`) |
| 디바이스 | GPU(CUDA) 우선, CPU fallback. lifespan 시점에 모델을 GPU에 상주시킴 |

---

## 2. 기술 스택 (AI / Framework)

### 2.1 런타임
- **Python**: `>=3.9` (`pyproject.toml`)
- **Docker base**: `pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime`
- **CUDA**: 12.1, cuDNN 8
- **OS 의존성**: `ffmpeg` (오디오 디코딩/리샘플링)

### 2.2 AI / Audio 라이브러리

| 라이브러리 | 버전 | 용도 |
|-----------|------|------|
| `torch` | ≥ 2.1 | 추론 백엔드 |
| `torchaudio` | ≥ 2.1 | WAV 로드/리샘플 |
| `torchvision` | ≥ 0.16 | (modelscope 의존성) |
| `modelscope` | ≥ 1.34 | ERes2Net 파이프라인 (`pipeline(task="speaker-verification")`) |
| `huggingface-hub` | ≥ 1.3.3 | 호환 의존성 |
| `librosa` | ≥ 0.11 | 보조 오디오 처리 |
| `soundfile` | ≥ 0.13 | WAV 입출력 |
| `numpy` | ≥ 1.24 (<2 Docker) / ≥ 2.0 (로컬) | 수치 연산 |
| `pandas` | ≥ 2.3 | 표 처리 |
| `kiwipiepy` | ≥ 0.22 | 한국어 형태소(텍스트 후처리) |
| `datasets` | < 3.0 | modelscope 호환 |
| `addict`, `simplejson`, `sortedcontainers`, `oss2`, `aiohttp` | - | modelscope 런타임 의존성 |

### 2.3 API / 서버
- **FastAPI** ≥ 0.128 — REST API
- **Uvicorn** ≥ 0.39 — ASGI 서버
- **python-multipart** ≥ 0.0.20 — 멀티파트 업로드

### 2.4 패키지 매니저
- 로컬: **uv** (`uv.lock`, `pyproject.toml`)
- Docker: `pip install --no-cache-dir <pinned list>` (재현성을 위해 명시적 핀)

---

## 3. 디렉토리 구조

```
speech_recognize/
├── Dockerfile
├── pyproject.toml
├── uv.lock
├── README.md
├── src/
│   ├── api.py                       # FastAPI 엔트리포인트 + lifespan 모델 프리로드
│   ├── v1/
│   │   ├── main.py                  # SpeakerEngine (ERes2Net wrapper)
│   │   ├── router.py                # /v1/recognize, /v1/jobs/{id}, /v1/refine-json
│   │   └── utils/
│   │       ├── job.py               # Job 관리(메모리 기반)
│   │       └── json_paser.py        # Whisper JSON 정규화 / segments 추출
│   └── resoursces/                  # ★ (오타이지만 코드 경로 그대로)
│       ├── models/
│       │   └── iic/speech_eres2net_base_sv_zh-cn_3dspeaker_16k/
│       └── employee/                # 사내 직원 enrollment 음성 (16k mono wav 권장)
│           ├── 홍길동/*.wav
│           └── 김기정/*.wav
└── docs/                            # 본 문서 위치
```

> **주의**: 폴더명이 `resoursces` 로 되어 있습니다(오타). 코드 경로가 이를 참조하므로 **리네이밍 금지**.

---

## 4. 환경변수

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `APP_PORT` | 8016 (코드 fallback 6003) | 서버 포트 (Docker `CMD` 가 사용) |
| `SPEAKER_MODEL_PATH` | `/app/src/resoursces/models/iic/speech_eres2net_base_sv_zh-cn_3dspeaker_16k` | ERes2Net 모델 경로 (Docker 기본값) |
| `EMPLOYEE_DB_PATH` | `<src>/resoursces/employee` | enrollment 음성 디렉토리 |
| `MS_CACHE_HOME` | 자동 설정(`SpeakerEngine.__init__`) | ModelScope 캐시 |

> **포트 규칙**: dev = 6016 / stg = 9016 / prd = 8016 (서비스 등록 시 `SPEECH_SERVICE_PORT`).

---

## 5. 로컬 개발 환경 구축

### 5.1 사전 요구사항
- NVIDIA GPU + 드라이버
- CUDA 12.1 호환
- `ffmpeg` (`apt-get install ffmpeg`)
- `uv` 또는 `pip`

### 5.2 설치 & 실행

```bash
cd /home/pps-nipa/jenkins/dev/speech_recognize

# (Conda 환경 사용 시)
conda create -n speech_recognize python=3.10 -y
conda activate speech_recognize

# 의존성 설치
uv sync                         # 또는: uv pip install -r requirements.txt

# 모델 준비
#   - ModelScope에서 iic/speech_eres2net_base_sv_zh-cn_3dspeaker_16k 다운로드
#   - src/resoursces/models/iic/... 에 배치

# 직원 enrollment 음성 준비
#   - src/resoursces/employee/<name>/<id>.wav (16kHz mono 권장. 자동 변환됨)

# 서버 실행
uv run uvicorn src.api:app --host 0.0.0.0 --port 8016 --reload
```

### 5.3 헬스체크
```bash
curl http://localhost:8016/health
# {"status":"healthy"}
```

---

## 6. Docker 실행

### 6.1 단독 실행 (Podman/Docker)
```bash
cd /home/pps-nipa/jenkins/dev/speech_recognize
docker build -t pps/speech-recognize:dev .

docker run --rm \
  --gpus all \
  -e APP_PORT=6016 \
  -p 6016:6016 \
  -v $(pwd)/src/resoursces/models:/app/src/resoursces/models:ro \
  -v $(pwd)/src/resoursces/employee:/app/src/resoursces/employee:ro \
  pps/speech-recognize:dev
```

> Podman + nvidia.com/gpu device 도 동일 패턴: `--device nvidia.com/gpu=all`.

### 6.2 Jenkins 통합 배포
- 이미지 태그: `IMG_SPEECH_SR=<registry>/speech-recognize:<env>`
- compose 서비스명: `speech_recognize`
- 내부 베이스 URL: `SPEECH_SR_BASE=http://speech_recognize:8016`

```bash
sudo /home/pps-nipa/jenkins/dev/docker.sh dev up speech_recognize
```

---

## 7. API 사용법

### 7.1 화자 식별 (비동기 Job)

**Step 1) Job 생성**
```bash
curl -X POST "http://localhost:8016/v1/recognize" \
  -H "Content-Type: multipart/form-data" \
  -F "audio=@meeting.wav" \
  -F "whisper_json=@whisper_output.json" \
  -F "threshold=0.25" \
  -F "mic_output_json=@mic_output.json"   # 선택
# -> {"job_id":"<uuid>","status":"pending"}
```

**Step 2) 상태 폴링**
```bash
curl "http://localhost:8016/v1/jobs/<uuid>"
```

응답(`JobInfo`)에 `progress`, `status: pending|processing|completed|failed`, `result.results` 가 포함됩니다.

### 7.2 Whisper JSON 정제 (보조)
```bash
curl -X POST "http://localhost:8016/v1/refine-json" \
  -F "whisper_json=@whisper_output.json"
```

마침표 단위로 세그먼트를 다시 묶고 시간을 재정렬합니다(`utils/json_paser.refine_whisper_json`).

### 7.3 Swagger
- `http://localhost:8016/docs`

---

## 8. 입력 데이터 규약

### 8.1 직원 enrollment 디렉토리
```
src/resoursces/employee/
├── 홍길동/
│   ├── sample1.wav
│   └── sample2.wav
└── 김기정/
    └── sample1.wav
```
- WAV 권장. 자동으로 16k mono 로 변환합니다(`SpeakerEngine.ensure_mono_16k`).
- 폴더명 = 화자 라벨.

### 8.2 Whisper JSON 입력
다음 중 하나의 구조가 허용됩니다.
1. `{"chunks": [...]}`
2. `{"segments": [...]}`
3. `{"result": {"segments": [...]}}`
4. 단순 리스트 `[{...}, {...}]`

각 세그먼트는 최소 `start`, `end`, `text` 필드를 가져야 합니다.

### 8.3 mic_output_json (선택)
`mic_speech_recognize` 의 RMS 기반 화자 구간 결과. 제공 시 ERes2Net 결과의 `speaker` 라벨을 사람 이름으로 매핑합니다.

---

## 9. 트러블슈팅

| 증상 | 원인 | 조치 |
|------|------|------|
| `Employee DB path not found` | `EMPLOYEE_DB_PATH` 잘못 또는 마운트 누락 | 호스트 경로/볼륨 확인 |
| 0.0 점수 / 결과 비정상 | enrollment 음성 부족/품질 저하 | 화자당 5~10초 이상의 깨끗한 발화 추가 |
| `pipeline` 로드 실패 | ModelScope 캐시 미설정 | `MS_CACHE_HOME` 확인, 모델 경로 절대경로로 |
| GPU OOM | 동시 작업 과다 | `BackgroundTasks` 동시성 제한, 또는 GPU 분리 |
| UTF-8 디코드 오류 (JSON) | 잘못된 파일 업로드 | Whisper JSON 인코딩 검증 |

---

## 10. 관련 문서
- 개발 표준 → [`./development-standards.md`](./development-standards.md)
- STT(음성→텍스트) → [`/home/pps-nipa/jenkins/dev/stt/docs/development-environment.md`](../../stt/docs/development-environment.md)
- Jenkins 배포 → [`/home/pps-nipa/jenkins/docs/development-environment.md`](../../../docs/development-environment.md)
