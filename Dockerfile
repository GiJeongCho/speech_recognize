FROM docker.io/pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
RUN pip install --no-cache-dir \
    "addict>=2.4.0" \
    "aiohttp>=3.13.3" \
    "datasets<3.0.0" \
    "fastapi>=0.128.0" \
    "huggingface-hub>=1.3.3" \
    "kiwipiepy>=0.22.2" \
    "librosa>=0.11.0" \
    "modelscope>=1.34.0" \
    "numpy>=1.24.0,<2" \
    "oss2>=2.19.1" \
    "pandas>=2.3.3" \
    "pillow>=11.3.0" \
    "python-multipart>=0.0.20" \
    "simplejson>=3.20.2" \
    "sortedcontainers>=2.4.0" \
    "soundfile>=0.13.1" \
    "torchaudio>=2.1.0" \
    "tqdm>=4.67.1" \
    "uvicorn>=0.39.0"

COPY src/ ./src/
COPY README.md ./

ENV PYTHONPATH=/app
ENV SPEAKER_MODEL_PATH=/app/src/resoursces/models/iic/speech_eres2net_base_sv_zh-cn_3dspeaker_16k

ENV APP_PORT=6003
EXPOSE ${APP_PORT}
CMD python -m uvicorn src.api:app --host 0.0.0.0 --port ${APP_PORT}
