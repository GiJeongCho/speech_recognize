import os
import torch
import torchaudio
import logging
import time
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Any, Union, Callable, Optional
from modelscope.pipelines import pipeline
from .utils.json_paser import refine_whisper_json

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
        whisper_data: Union[Dict, List[Dict]], 
        speakers_root: str, 
        threshold: float = 0.1,
        progress_callback: Optional[Callable[[float], None]] = None
    ) -> Dict:
        start_time = time.time()
        
        # 초기 진행률 보고 (작업 시작)
        if progress_callback:
            progress_callback(1.0)

        # 1. 원본 오디오 로드 및 전처리
        wav, sr = torchaudio.load(full_audio_path)
        wav = self.ensure_mono_16k(wav, sr)
        sr = 16000
        n_samples = wav.size(1)

        # 2. 기준 화자(Enrollment) 파일 목록 확보 및 전처리 (16k mono WAV로 변환)
        # 이 과정을 전체 공정의 약 10%로 간주
        speakers_path = Path(speakers_root)
        enroll_data = {}
        temp_enroll_dir = Path(f"/tmp/enroll_{int(time.time())}_{os.getpid()}")
        temp_enroll_dir.mkdir(parents=True, exist_ok=True)
        
        if progress_callback:
            progress_callback(5.0)

        try:
            if speakers_path.exists():
                # speakers_root 아래의 각 디렉토리를 화자로 간주
                for spk_dir in sorted([p for p in speakers_path.iterdir() if p.is_dir()]):
                    spk_name = spk_dir.name
                    refs = []
                    for ext in [".wav", ".flac", ".m4a", ".mp3", ".WAV", ".FLAC", ".M4A", ".MP3"]:
                        for f in spk_dir.glob(f"*{ext}"):
                            try:
                                # 16k mono WAV로 변환하여 임시 저장
                                wav_enroll, sr_enroll = torchaudio.load(str(f))
                                wav_enroll = self.ensure_mono_16k(wav_enroll, sr_enroll)
                                tmp_f = temp_enroll_dir / f"{spk_name}_{f.stem}.wav"
                                torchaudio.save(str(tmp_f), wav_enroll, 16000)
                                refs.append(tmp_f)
                            except Exception as e:
                                logger.error(f"Failed to process enrollment file {f}: {e}")
                    
                    if refs:
                        enroll_data[spk_name] = refs

                # speakers_root 자체에 오디오 파일이 있는 경우 (화자 이름은 파일명)
                for ext in [".wav", ".flac", ".m4a", ".mp3", ".WAV", ".FLAC", ".M4A", ".MP3"]:
                    for f in speakers_path.glob(f"*{ext}"):
                        if f.stem not in enroll_data:
                            try:
                                wav_enroll, sr_enroll = torchaudio.load(str(f))
                                wav_enroll = self.ensure_mono_16k(wav_enroll, sr_enroll)
                                tmp_f = temp_enroll_dir / f"direct_{f.stem}.wav"
                                torchaudio.save(str(tmp_f), wav_enroll, 16000)
                                enroll_data[f.stem] = [tmp_f]
                            except Exception as e:
                                logger.error(f"Failed to process direct enrollment file {f}: {e}")

            if not enroll_data:
                logger.error(f"No speaker enrollment files found in {speakers_root}")
                raise RuntimeError(f"No speaker enrollment files found in {speakers_root}")

            logger.info(f"Loaded {len(enroll_data)} speakers for identification")
            
            # Enrollment 완료 시점 (10%)
            if progress_callback:
                progress_callback(10.0)

            # 3. 각 청크별 화자 비교 (refine_whisper_json 유틸 사용)
            results = []
            temp_seg_path = f"/tmp/seg_{int(time.time())}_{os.getpid()}.wav"

            # 외부 유틸리티를 사용하여 문장 단위로 재구성
            final_chunks = refine_whisper_json(whisper_data)
            total_chunks = len(final_chunks)

            # 확정된 문장 단위 청크들에 대해 화자 식별 수행
            for i, chunk in enumerate(final_chunks):
                start, end = chunk["start"], chunk["end"]
                original_speaker = chunk.get("speaker", "unknown")
                
                # 청크 잘라내기
                s_idx = max(0, int(round(start * sr)))
                e_idx = min(n_samples, int(round(end * sr)))
                
                if e_idx <= s_idx:
                    continue
                    
                seg_wav = wav[:, s_idx:e_idx]

                # 0.5초 미만의 짧은 오디오는 복제하여 길이를 늘림 (화자 식별 정확도 향상)
                # 0.2초 이하 구간은 이미 json_parser에서 병합되었으므로, 여기서는 0.2~0.5초 구간이 주 대상
                seg_duration = seg_wav.size(1) / sr
                if seg_duration > 0 and seg_duration < 0.5:
                    # 0.5초 이상이 될 때까지 반복 (최소 2배)
                    repeat_count = int(0.5 / seg_duration) + 1
                    seg_wav = seg_wav.repeat(1, repeat_count)

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
                            # 예외 발생 시 로거를 사용하여 기록 [[memory:6804125]]
                            logger.error(f"Error comparing {temp_seg_path} with {ref_path}: {e}")
                    
                    if spk_best > best_score:
                        best_score = spk_best
                        best_spk = spk_name
                
                assigned = best_spk if best_score >= threshold else "unknown"
                results.append({
                    "start": round(start, 3),
                    "end": round(end, 3),
                    "text": chunk["text"],
                    "speaker": assigned,
                    "original_speaker": original_speaker,  # 후처리를 위해 원본 화자 정보 임시 저장
                    "score": round(float(best_score), 4) if best_score != -1.0 else 0.0
                })

                # 진행률 업데이트 (10% ~ 99%)
                # 남은 90%를 청크 개수로 나누어 할당
                if progress_callback and total_chunks > 0:
                    current_percent = 10.0 + (float(i + 1) / total_chunks * 89.0)
                    progress_callback(current_percent)

            # --- 후처리: 화자 일관성 보정 (80% 룰) ---
            from collections import defaultdict
            
            # 1. 통계 집계: 원본 화자별로 식별된 화자 카운트
            speaker_stats = defaultdict(lambda: {"total": 0, "counts": defaultdict(int)})
            
            for r in results:
                orig = r.get("original_speaker")
                identified = r["speaker"]
                if not orig: continue
                
                speaker_stats[orig]["total"] += 1
                if identified != "unknown":
                    speaker_stats[orig]["counts"][identified] += 1
            
            # 2. 매핑 규칙 생성
            mapping = {}
            for orig, stats in speaker_stats.items():
                total = stats["total"]
                if total == 0:
                    mapping[orig] = orig
                    continue
                
                # 가장 많이 식별된 화자 찾기
                best_identified = None
                max_count = 0
                for name, count in stats["counts"].items():
                    if count > max_count:
                        max_count = count
                        best_identified = name
                
                # 해당 화자가 전체의 80% 이상을 차지하면, 나머지도 그 사람으로 간주
                if best_identified and (max_count / total >= 0.8):
                    mapping[orig] = best_identified
                else:
                    # 그렇지 않으면 원본 화자(SPEAKER_XX) 사용 (외부 화자 처리)
                    mapping[orig] = orig
            
            # 3. 결과 업데이트 (unknown인 경우에만 매핑 적용)
            for r in results:
                if r["speaker"] == "unknown":
                    orig = r.get("original_speaker")
                    if orig:
                        r["speaker"] = mapping.get(orig, orig)
                
                # 임시 필드 제거
                if "original_speaker" in r:
                    del r["original_speaker"]
            # --------------------------------------

        finally:
            # 임시 파일 및 폴더 정리
            if temp_enroll_dir.exists():
                shutil.rmtree(temp_enroll_dir)
            if 'temp_seg_path' in locals() and os.path.exists(temp_seg_path):
                os.remove(temp_seg_path)

        end_time = time.time()
        
        # 완료 시 100% 호출은 하지 않음 (외부 JobManager에서 완료 처리 시 수행)
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
