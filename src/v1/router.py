from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from typing import List, Dict, Any
import json
import os
import uuid
import shutil
import logging
from .main import get_engine
from .utils.json_paser import refine_whisper_json
from .utils.job import job_manager, JobInfo

logger = logging.getLogger(__name__)

router_v1 = APIRouter(prefix="/v1", tags=["speaker"])

# 사내 직원 목소리 DB 경로 설정
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_EMPLOYEE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "resoursces", "employee"))

def _background_recognition_task(
    job_id: str, 
    audio_path: str, 
    json_path: str, 
    threshold: float,
    speakers_path: str
):
    """
    백그라운드에서 실행되는 화자 인식 작업 함수
    """
    try:
        # 진행률 업데이트를 위한 콜백 함수
        def update_progress(p: float):
            job_manager.update_progress(job_id, p)

        # 1. Whisper JSON 읽기
        with open(json_path, "r", encoding="utf-8") as f:
            whisper_data = json.load(f)
            
        # chunks(Whisper) 또는 segments(WhisperX) 키가 있는지 확인
        if isinstance(whisper_data, dict):
            valid_data = whisper_data.get("chunks") or whisper_data.get("segments")
            if valid_data is None:
                raise ValueError("No 'chunks' or 'segments' found in Whisper JSON")
        elif isinstance(whisper_data, list):
            valid_data = whisper_data # list itself is likely segments
        else:
            raise ValueError("Invalid Whisper JSON format")

        # 2. 화자 인식 실행
        engine = get_engine()
        result = engine.identify_speaker(
            full_audio_path=audio_path, 
            whisper_data=whisper_data, 
            speakers_root=speakers_path, 
            threshold=threshold,
            progress_callback=update_progress
        )

        # 3. 작업 완료 처리
        job_manager.complete_job(job_id, result)
        logger.info(f"Job {job_id} completed successfully.")

    except Exception as e:
        logger.exception(f"Job {job_id} failed: {e}") # [[memory:6804125]]
        job_manager.fail_job(job_id, str(e))
        
    finally:
        # 임시 파일 정리 (작업 종료 후)
        for p in [audio_path, json_path]:
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception as cleanup_err:
                logger.warning(f"Failed to cleanup temp file {p}: {cleanup_err}")

@router_v1.post("/recognize", response_model=Dict[str, str])
async def recognize_speaker(
    background_tasks: BackgroundTasks,
    audio: UploadFile = File(..., description="화자를 식별할 원본 음성 파일 (wav, mp3, m4a)"),
    whisper_json: UploadFile = File(..., description="Whisper STT 결과 JSON 파일 (chunks 포함)"),
    threshold: float = Form(0.2, description="화자 일치 여부를 판단할 임계값")
):
    """
    화자 인식 작업을 시작하고 Job ID를 반환합니다.
    진행 상황은 GET /v1/jobs/{job_id} 로 확인할 수 있습니다.
    """
    # 1. Job ID 생성
    job_id = str(uuid.uuid4())
    job_manager.create_job(job_id)

    # 2. 파일 임시 저장 (Background Task에서 접근 가능하도록)
    # /tmp 디렉토리에 job_id를 prefix로 사용하여 저장
    temp_audio = f"/tmp/{job_id}_{audio.filename}"
    temp_json = f"/tmp/{job_id}_{whisper_json.filename}"

    try:
        with open(temp_audio, "wb") as buffer:
            shutil.copyfileobj(audio.file, buffer)
        with open(temp_json, "wb") as buffer:
            shutil.copyfileobj(whisper_json.file, buffer)
    except Exception as e:
        logger.exception(f"Failed to save upload files: {e}")
        job_manager.fail_job(job_id, f"File upload failed: {e}")
        raise HTTPException(status_code=500, detail="File upload failed")

    # 3. 사내 직원 DB 경로 확인
    target_speakers_path = os.getenv("EMPLOYEE_DB_PATH", DEFAULT_EMPLOYEE_DIR)
    if not os.path.exists(target_speakers_path):
        # 파일은 삭제하고 에러 처리
        if os.path.exists(temp_audio): os.remove(temp_audio)
        if os.path.exists(temp_json): os.remove(temp_json)
        raise HTTPException(status_code=500, detail=f"Employee DB path not found: {target_speakers_path}")

    # 4. 백그라운드 작업 등록
    background_tasks.add_task(
        _background_recognition_task,
        job_id,
        temp_audio,
        temp_json,
        threshold,
        target_speakers_path
    )

    return {"job_id": job_id, "status": "pending"}

@router_v1.get("/jobs/{job_id}", response_model=JobInfo)
async def get_job_status(job_id: str):
    """
    특정 Job의 진행 상태 및 결과를 조회합니다.
    """
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job

@router_v1.post("/refine-json")
async def refine_json(
    whisper_json: UploadFile = File(..., description="정제할 Whisper STT 결과 JSON 파일")
):
    """
    업로드된 Whisper JSON을 마침표 단위 문장으로 묶고 시간을 재조정하여 반환합니다.
    """
    try:
        content = await whisper_json.read()
        whisper_data = json.loads(content)
        
        refined_data = refine_whisper_json(whisper_data)
        return {
            "status": "success",
            "count": len(refined_data),
            "results": refined_data
        }
    except Exception as e:
        logger.exception(f"Error in refine_json: {e}")
        raise HTTPException(status_code=500, detail=str(e))
