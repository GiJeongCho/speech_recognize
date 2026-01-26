# 1. 빌드 스테이지
FROM ghcr.io/astral-sh/uv:python3.9-bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
WORKDIR /app

# 시스템 라이브러리 (ffmpeg)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# 의존성 파일 복사 및 설치
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

# 2. 실행 스테이지
# H100(sm_90) 호환성을 위해 CUDA 12.1 이상 이미지를 사용합니다.
FROM docker.io/pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# 빌드 스테이지에서 생성된 가상환경 복사
COPY --from=builder /app/.venv /app/.venv
COPY src/ ./src/
COPY README.md ./

# 시스템 라이브러리 설치 (ffmpeg는 런타임에도 필요)
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

# 환경 변수 설정
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH=/app
ENV SPEAKER_MODEL_PATH=/app/src/resoursces/models/iic/speech_eres2net_base_sv_zh-cn_3dspeaker_16k

# API 실행 (가상환경의 uvicorn 사용)
EXPOSE 8011
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8011"]
