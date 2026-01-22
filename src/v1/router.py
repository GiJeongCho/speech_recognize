from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import List
import json
import os
import uuid
import shutil
from .main import get_engine

router_v1 = APIRouter(prefix="/v1", tags=["speaker"])

# 사내 직원 목소리 DB 경로 설정
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_EMPLOYEE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "resoursces", "employee"))

@router_v1.post("/recognize")
async def recognize_speaker(
    audio: UploadFile = File(..., description="분석할 메인 오디오 파일 (WAV)"),
    whisper_json: UploadFile = File(..., description="Whisper STT 결과 JSON 파일"),
    threshold: float = Form(0.25, description="화자 일치 임계값 (기본 0.25)")
):
    # 임시 파일 저장
    temp_id = str(uuid.uuid4())
    temp_audio = f"/tmp/{temp_id}_{audio.filename}"
    temp_json = f"/tmp/{temp_id}_{whisper_json.filename}"

    try:
        # 메인 오디오 및 JSON 저장
        with open(temp_audio, "wb") as buffer:
            shutil.copyfileobj(audio.file, buffer)
        with open(temp_json, "wb") as buffer:
            shutil.copyfileobj(whisper_json.file, buffer)

        # 사내 직원 DB 경로 사용
        target_speakers_path = os.getenv("EMPLOYEE_DB_PATH", DEFAULT_EMPLOYEE_DIR)
        
        if not os.path.exists(target_speakers_path):
            raise HTTPException(status_code=500, detail=f"Employee DB path not found: {target_speakers_path}")

        # Whisper JSON 읽기
        with open(temp_json, "r", encoding="utf-8") as f:
            whisper_data = json.load(f)
            chunks = whisper_data.get("chunks", [])

        if not chunks:
            raise HTTPException(status_code=400, detail="No chunks found in Whisper JSON")

        # 화자 인식 실행
        engine = get_engine()
        result = engine.identify_speaker(
            temp_audio, 
            chunks, 
            target_speakers_path, 
            threshold=threshold
        )

        return result

    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.exception(f"Error in recognize_speaker: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # 임시 파일 정리
        for p in [temp_audio, temp_json]:
            if os.path.exists(p):
                os.remove(p)
