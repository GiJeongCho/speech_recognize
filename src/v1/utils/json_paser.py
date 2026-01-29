import re
from typing import Dict, List, Any
from .kr_tag import kiwi_tagger

def refine_whisper_json(whisper_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Kiwi의 어미(EF, EC) 분석을 사용하여 문장을 나눕니다.
    추가적인 병합 로직 없이 어미가 발견되는 즉시 분리하여 1문장 단위를 유지합니다.
    """
    raw_segments = whisper_data.get("segments") or whisper_data.get("chunks") or []
    
    final_chunks = []
    current_words = []
    
    for seg in raw_segments:
        words = seg.get("words", [])
        if not words:
            final_chunks.append({
                "start": float(seg.get("start", 0)),
                "end": float(seg.get("end", 0)),
                "text": seg.get("text", "").strip()
            })
            continue
            
        for w in words:
            current_words.append(w)
            w_text = (w.get("word") or w.get("text", "")).strip()
            
            # 어미(EF, EC)가 포함된 단어면 즉시 문장으로 확정하고 분리
            if kiwi_tagger.is_terminal_ending(w_text):
                final_chunks.append({
                    "start": float(current_words[0].get("start", 0)),
                    "end": float(current_words[-1].get("end", 0)),
                    "text": " ".join([(wd.get("word") or wd.get("text", "")).strip() for wd in current_words])
                })
                current_words = []
                
    # 마지막 남은 단어들 처리
    if current_words:
        final_chunks.append({
            "start": float(current_words[0].get("start", 0)),
            "end": float(current_words[-1].get("end", 0)),
            "text": " ".join([(wd.get("word") or wd.get("text", "")).strip() for wd in current_words])
        })

    return final_chunks
