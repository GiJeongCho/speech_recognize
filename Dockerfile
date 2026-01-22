# 1. 빌드 스테이지
FROM ghcr.io/astral-sh/uv:python3.10-bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
WORKDIR /app

# 시스템 라이브러리 (ffmpeg)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

# 2. 실행 스테이지
FROM docker.io/pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
COPY src/ ./src/

# 시스템 라이브러리 설치
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

# 환경 변수 설정
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH=/app
ENV SPEAKER_MODEL_PATH=/app/src/resoursces/models/iic/speech_eres2net_base_sv_zh-cn_3dspeaker_16k

# API 실행
CMD ["uv", "run", "uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8020"]
