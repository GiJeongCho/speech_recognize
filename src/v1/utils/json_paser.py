import re
from typing import Dict, List, Any
from .kr_tag import kiwi_tagger

def refine_whisper_json(whisper_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    정제 로직 단계:
    1. 분할: EF는 무조건 분리, EC는 다음 단어와 간격이 0.5초 이상일 때 분리
    2. 보완 병합: 분리된 문장 중 1.0초 미만인 것은 앞 문장에 합쳐서 최소 분석 길이 확보
    """
    raw_segments = whisper_data.get("segments") or whisper_data.get("chunks") or []
    
    initial_chunks = []
    current_words = []
    
    # [1단계] 단어들을 가져와서 어미 규칙에 따라 1차 분할
    all_words = []
    for seg in raw_segments:
        words = seg.get("words", [])
        if not words:
            all_words.append({
                "start": float(seg.get("start", 0)),
                "end": float(seg.get("end", 0)),
                "word": seg.get("text", "").strip()
            })
        else:
            all_words.extend(words)

    for i, w in enumerate(all_words):
        current_words.append(w)
        w_text = (w.get("word") or w.get("text", "")).strip()
        ending_type = kiwi_tagger.get_ending_type(w_text)
        
        should_split = False
        if ending_type == 'EF':
            should_split = True
        elif ending_type == 'EC':
            if i + 1 < len(all_words):
                next_w = all_words[i+1]
                gap = float(next_w.get("start", 0)) - float(w.get("end", 0))
                if gap >= 0.5:
                    should_split = True
            else:
                should_split = True
        
        if should_split:
            initial_chunks.append({
                "start": float(current_words[0].get("start", 0)),
                "end": float(current_words[-1].get("end", 0)),
                "text": " ".join([(wd.get("word") or wd.get("text", "")).strip() for wd in current_words])
            })
            current_words = []
                
    if current_words:
        initial_chunks.append({
            "start": float(current_words[0].get("start", 0)),
            "end": float(current_words[-1].get("end", 0)),
            "text": " ".join([(wd.get("word") or wd.get("text", "")).strip() for wd in current_words])
        })

    # [2단계] 1.0초 미만의 짧은 문장은 이전 문장에 병합 (데이터 보존)
    final_results = []
    for chunk in initial_chunks:
        duration = chunk["end"] - chunk["start"]
        
        if duration < 1.0 and final_results:
            # 현재 문장이 1초 미만이면 이전 문장에 내용을 합치고 종료 시간만 업데이트
            final_results[-1]["end"] = chunk["end"]
            final_results[-1]["text"] += " " + chunk["text"]
        else:
            final_results.append(chunk)
            
    return final_results
