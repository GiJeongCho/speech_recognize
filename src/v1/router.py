from fastapi import APIRouter, UploadFile, File, Form, HTTPException
import json
import os
import uuid
import shutil
from .main import get_engine

router_v1 = APIRouter(prefix="/v1", tags=["speaker"])

@router_v1.post("/recognize")
async def recognize_speaker(
    audio: UploadFile = File(...),
    whisper_json: UploadFile = File(...),
    speakers_root: str = Form("/app/speaker_chunks"),
    threshold: float = Form(0.25)
):
    # 임시 파일 저장
    temp_id = str(uuid.uuid4())
    temp_audio = f"/tmp/{temp_id}_{audio.filename}"
    temp_json = f"/tmp/{temp_id}_{whisper_json.filename}"

    try:
        with open(temp_audio, "wb") as buffer:
            shutil.copyfileobj(audio.file, buffer)
        with open(temp_json, "wb") as buffer:
            shutil.copyfileobj(whisper_json.file, buffer)

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
            speakers_root, 
            threshold=threshold
        )

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # 임시 파일 정리
        for p in [temp_audio, temp_json]:
            if os.path.exists(p):
                os.remove(p)
