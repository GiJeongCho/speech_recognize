from contextlib import asynccontextmanager
from fastapi import FastAPI
import uvicorn
import logging
from src.v1.router import router_v1
from src.v1.main import get_engine

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 서버 시작 시 ERes2Net 모델을 즉시 로드하여 GPU에 항시 띄워둡니다.
    logger.info("Pre-loading ERes2Net model into GPU memory...")
    try:
        get_engine()
        logger.info("ERes2Net model is now resident in GPU memory.")
    except Exception as e:
        logger.error(f"Failed to pre-load ERes2Net model: {e}")
    yield

app = FastAPI(
    title="Speaker Recognition API",
    description="ERes2Net 기반 화자 식별 서비스 (Whisper JSON 연동)",
    version="1.0.0",
    lifespan=lifespan
)

app.include_router(router_v1)

@app.get("/health")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run("src.api:app", host="0.0.0.0", port=8011, reload=True)
