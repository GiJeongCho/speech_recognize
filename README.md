# 대본 (화자별 읽어서 DB에 저장할)
```
"다음 뉴스입니다. 인공지능 기술이 발달하면서 우리 생활 속 다양한 분야에 변화가 일어나고 있습니다. 특히 음성 인식 기술은 회의록 자동 작성부터 실시간 통역까지 그 활용 범위가 넓어지고 있습니다. 전문가들은 앞으로 개인 맞춤형 음성 서비스가 더욱 정교해질 것이라고 전망했습니다."
```

# Speaker Recognition API

`ERes2Net` (3D-Speaker) 모델을 사용하여 Whisper STT 결과의 각 구간이 어떤 화자의 목소리인지 식별해주는 API 서비스입니다.

## 주요 기능
- **사내 직원 식별**: 프로젝트 내부에 저장된 직원 음성 데이터(`src/resoursces/employee`)를 기반으로 화자 판별
- **Whisper 연동**: Whisper가 생성한 JSON (chunks 포함)을 입력받아 텍스트별 화자 할당
- **GPU 가속**: CUDA를 통한 빠른 화자 임베딩 추출 및 비교

## 실행 방법

### 1. 로컬 환경 실행
```bash
conda activate speech_recognize
# 의존성 설치
uv pip install -r requirements.txt
# 서버 실행
uv run uvicorn src.api:app --host 0.0.0.0 --port 8016 --reload
```

### 2. 컨테이너 환경 실행 (Podman/Docker)
```bash
# 이미지 빌드
podman build -t pps/speech_recognize:v0.0.1 -f Dockerfile .

# 컨테이너 실행
podman run -d \
  --name speaker-recognize \
  --device nvidia.com/gpu=all \
  -p 8016:8016 \
  -v /home/pps-nipa/NIQ/fish/speech_recognize/src/resoursces/models:/app/src/resoursces/models:Z \
  -v /home/pps-nipa/NIQ/fish/speech_recognize/src/resoursces/employee:/app/src/resoursces/employee:Z \
  pps/speech_recognize:v0.0.1
```

## API 사용법

### 화자 식별 (`POST /v1/recognize`)
음성 파일과 Whisper 결과를 전송하여 사내 직원 데이터베이스와 비교해 화자를 구분합니다.

- **Endpoint**: `http://localhost:8016/v1/recognize`
- **Parameters**:
  - `audio`: 분석할 메인 음성 파일 (WAV 권장)
  - `whisper_json`: Whisper STT 결과 JSON (chunks 리스트 포함)
  - `threshold`: 화자 일치 임계값 (기본값: `0.25`)

**cURL 테스트 예시:**
```bash
curl -X 'POST' \
  'http://localhost:8016/v1/recognize' \
  -H 'Content-Type: multipart/form-data' \
  -F 'audio=@meeting.wav' \
  -F 'whisper_json=@whisper_output.json' \
  -F 'threshold=0.25'
```

## API 문서 및 모니터링
- **Swagger UI**: [http://localhost:8016/docs](http://localhost:8016/docs)
- **Health Check**: [http://localhost:8016/health](http://localhost:8016/health)
