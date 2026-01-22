import os
import torch
import torchaudio
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any
from modelscope.pipelines import pipeline

logger = logging.getLogger(__name__)

class SpeakerEngine:
    def __init__(self, model_path: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading ERes2Net model from {model_path} on {self.device}")
        
        # 모델 저장 경로 및 임시 폴더 설정
        self.model_path = model_path
        os.environ["MS_CACHE_HOME"] = os.path.dirname(model_path)
        
        # 스피커 검증 파이프라인 로드
        self.sv_pipeline = pipeline(
            task="speaker-verification", 
            model=model_path, 
            device=self.device
        )
        logger.info("ERes2Net model is successfully pinned to GPU.")

    def ensure_mono_16k(self, wav: torch.Tensor, sr: int) -> torch.Tensor:
        # ... (생략)
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)
        return wav

    def extract_score(self, result: Any) -> float:
        # ... (생략)
        if isinstance(result, list) and result:
            return self.extract_score(result[0])
        return 0.0

    def identify_speaker(
        self, 
        full_audio_path: str, 
        whisper_chunks: List[Dict], 
        speakers_root: str, 
        threshold: float = 0.25
    ) -> Dict:
        start_time = time.time()
        # 1. 원본 오디오 로드 및 전처리
        wav, sr = torchaudio.load(full_audio_path)
        wav = self.ensure_mono_16k(wav, sr)
        sr = 16000

        # 2. 기준 화자(Enrollment) 파일 목록 확보
        speakers_path = Path(speakers_root)
        enroll_data = {}
        # ... (생략)
        if not enroll_data:
            raise RuntimeError(f"No speaker enrollment files found in {speakers_root}")

        # 3. 각 청크별 화자 비교
        results = []
        temp_seg_path = "/tmp/current_seg.wav"

        for i, chunk in enumerate(whisper_chunks):
            # ... (기존 루프 로직)
            assigned = best_spk if best_score >= threshold else "unknown"
            results.append({
                "start": start,
                "end": end,
                "text": chunk.get("text", ""),
                "speaker": assigned,
                "score": round(float(best_score), 4)
            })

        end_time = time.time()
        return {
            "status": "success",
            "processing_time": f"{round(end_time - start_time, 2)}s",
            "results": results
        }

# 싱글톤 관리
# 현재 파일(src/v1/main.py) 기준으로 모델 상대 경로 설정
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.abspath(os.path.join(CURRENT_DIR, "..", "resoursces", "models", "iic", "speech_eres2net_base_sv_zh-cn_3dspeaker_16k"))

MODEL_PATH = os.getenv("SPEAKER_MODEL_PATH", DEFAULT_MODEL_PATH)
engine = None

def get_engine():
    global engine
    if engine is None:
        engine = SpeakerEngine(MODEL_PATH)
    return engine
