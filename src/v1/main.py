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
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)

        target_sr = 16000
        if sr != target_sr:
            wav = torchaudio.functional.resample(wav, sr, target_sr)
        return wav

    def extract_score(self, result: Any) -> float:
        if isinstance(result, (float, int)):
            return float(result)
        if isinstance(result, dict):
            for k in ("score", "scores", "similarity", "cosine_score"):
                if k in result:
                    v = result[k]
                    if isinstance(v, (float, int)):
                        return float(v)
                    if isinstance(v, list) and v:
                        return float(v[0])
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
        n_samples = wav.size(1)

        # 2. 기준 화자(Enrollment) 파일 목록 확보
        speakers_path = Path(speakers_root)
        enroll_data = {}
        
        if speakers_path.exists():
            # speakers_root 아래의 각 디렉토리를 화자로 간주
            for spk_dir in sorted([p for p in speakers_path.iterdir() if p.is_dir()]):
                spk_name = spk_dir.name
                refs = []
                for ext in [".wav", ".flac", ".m4a", ".mp3"]:
                    refs.extend(list(spk_dir.glob(f"*{ext}")))
                    refs.extend(list(spk_dir.glob(f"*{ext.upper()}")))
                
                if refs:
                    enroll_data[spk_name] = refs

            # speakers_root 자체에 오디오 파일이 있는 경우 (화자 이름은 파일명)
            for ext in [".wav", ".flac", ".m4a", ".mp3"]:
                for f in speakers_path.glob(f"*{ext}"):
                    if f.stem not in enroll_data:
                        enroll_data[f.stem] = [f]
                for f in speakers_path.glob(f"*{ext.upper()}"):
                    if f.stem not in enroll_data:
                        enroll_data[f.stem] = [f]

        if not enroll_data:
            logger.error(f"No speaker enrollment files found in {speakers_root}")
            raise RuntimeError(f"No speaker enrollment files found in {speakers_root}")

        # 3. 각 청크별 화자 비교
        results = []
        temp_seg_path = f"/tmp/seg_{int(time.time())}.wav"

        try:
            for i, chunk in enumerate(whisper_chunks):
                start = float(chunk.get("start", 0))
                end = float(chunk.get("end", 0))
                
                # 청크 잘라내기
                s_idx = max(0, int(round(start * sr)))
                e_idx = min(n_samples, int(round(end * sr)))
                
                if e_idx <= s_idx:
                    continue
                    
                seg_wav = wav[:, s_idx:e_idx]
                torchaudio.save(temp_seg_path, seg_wav, sr)
                
                best_spk = "unknown"
                best_score = -1.0
                
                for spk_name, refs in enroll_data.items():
                    spk_best = -1.0
                    for ref_path in refs:
                        try:
                            r = self.sv_pipeline([temp_seg_path, str(ref_path)])
                            score = self.extract_score(r)
                            if score > spk_best:
                                spk_best = score
                        except Exception as e:
                            logger.error(f"Error comparing {temp_seg_path} with {ref_path}: {e}")
                    
                    if spk_best > best_score:
                        best_score = spk_best
                        best_spk = spk_name
                
                assigned = best_spk if best_score >= threshold else "unknown"
                results.append({
                    "start": start,
                    "end": end,
                    "text": chunk.get("text", ""),
                    "speaker": assigned,
                    "score": round(float(best_score), 4)
                })
        finally:
            if os.path.exists(temp_seg_path):
                os.remove(temp_seg_path)

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
