from typing import Dict, List, Any
from .kr_tag import kiwi_tagger

def refine_whisper_json(whisper_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    WhisperX 또는 일반 Whisper 결과 JSON을 분석하여 
    Kiwi의 종결 어미(EF)를 기준으로 문장을 재구성합니다.
    """
    raw_segments = whisper_data.get("segments") or whisper_data.get("chunks") or []
    
    # [단계 1] 마침표 또는 종결 어미 단위로 기초 문장 생성 (데이터 100% 보존)
    all_sentences = []
    current_words = []
    
    for seg in raw_segments:
        words = seg.get("words", [])
        if not words:
            # 단어 정보가 없는 경우 세그먼트 데이터 그대로 사용
            all_sentences.append({
                "start": float(seg.get("start", 0)),
                "end": float(seg.get("end", 0)),
                "text": seg.get("text", "").strip()
            })
            continue
        
        for w in words:
            w_text = (w.get("word") or w.get("text", "")).strip()
            if not w_text: continue
            
            current_words.append(w)
            
            # KiwiTagger 클래스를 사용하여 종결 어미(EF) 여부 확인
            if kiwi_tagger.is_terminal_ending(w_text):
                all_sentences.append({
                    "start": float(current_words[0].get("start", 0)),
                    "end": float(current_words[-1].get("end", 0)),
                    "text": " ".join([(wd.get("word") or wd.get("text", "")).strip() for wd in current_words])
                })
                current_words = []
    
    # 남은 단어들 처리
    if current_words:
        all_sentences.append({
            "start": float(current_words[0].get("start", 0)),
            "end": float(current_words[-1].get("end", 0)),
            "text": " ".join([(wd.get("word") or wd.get("text", "")).strip() for wd in current_words])
        })

    # [단계 2] 모델 분석을 위해 1.0초 미만 문장은 주변 문장과 병합
    final_chunks = []
    if all_sentences:
        temp_c = all_sentences[0]
        for next_c in all_sentences[1:]:
            duration = temp_c["end"] - temp_c["start"]
            # 1.0초 미만인 경우 누락 없이 다음 문장에 병합
            if duration < 1.0:
                temp_c["end"] = next_c["end"]
                temp_c["text"] += " " + next_c["text"]
            else:
                final_chunks.append(temp_c)
                temp_c = next_c
        final_chunks.append(temp_c)
    
    return final_chunks
