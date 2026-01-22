from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import List
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
    speaker_files: List[UploadFile] = File(None),
    speakers_root: str = Form("/app/speaker_chunks"),
    threshold: float = Form(0.25)
):
    # 임시 파일 저장
    temp_id = str(uuid.uuid4())
    temp_audio = f"/tmp/{temp_id}_{audio.filename}"
    temp_json = f"/tmp/{temp_id}_{whisper_json.filename}"
    temp_speaker_dir = f"/tmp/{temp_id}_speakers"

    try:
        with open(temp_audio, "wb") as buffer:
            shutil.copyfileobj(audio.file, buffer)
        with open(temp_json, "wb") as buffer:
            shutil.copyfileobj(whisper_json.file, buffer)

        # 업로드된 화자 파일이 있으면 임시 디렉토리에 저장
        if speaker_files:
            os.makedirs(temp_speaker_dir, exist_ok=True)
            for spk_file in speaker_files:
                spk_path = os.path.join(temp_speaker_dir, spk_file.filename)
                with open(spk_path, "wb") as buffer:
                    shutil.copyfileobj(spk_file.file, buffer)
            # 업로드된 파일들을 우선적으로 사용
            speakers_root = temp_speaker_dir

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
        # 임시 파일 및 디렉토리 정리
        for p in [temp_audio, temp_json]:
            if os.path.exists(p):
                os.remove(p)
        if os.path.exists(temp_speaker_dir):
            shutil.rmtree(temp_speaker_dir)
