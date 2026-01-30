import re
import logging
from typing import Dict, List, Any, Optional
from .kr_tag import kiwi_tagger

logger = logging.getLogger(__name__)

def refine_whisper_json(whisper_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    정제 로직 단계:
    1. 분할: EF는 무조건 분리, EC는 다음 단어와 간격이 0.5초 이상일 때 분리, 화자가 바뀌면 분리
    2. 보완 병합: 분리된 문장 중 1.0초 미만인 것은 '화자가 같을 때만' 앞 문장에 합쳐서 최소 분석 길이 확보
    """
    raw_segments = whisper_data.get("segments") or whisper_data.get("chunks") or []
    
    initial_chunks = []
    current_words = []
    
    # [1단계] 단어들을 가져와서 어미 규칙 및 화자 변경에 따라 1차 분할
    all_words = []
    for seg in raw_segments:
        words = seg.get("words", [])
        if not words:
            # 단어 단위 정보가 없는 경우 세그먼트 자체를 하나의 단어처럼 처리
            all_words.append({
                "start": float(seg.get("start", 0)),
                "end": float(seg.get("end", 0)),
                "word": seg.get("text", "").strip(),
                "speaker": seg.get("speaker")
            })
        else:
            all_words.extend(words)

    for i, w in enumerate(all_words):
        current_words.append(w)
        w_text = (w.get("word") or w.get("text", "")).strip()
        current_speaker = w.get("speaker")
        
        should_split = False
        
        # 다음 단어 확인
        next_w = all_words[i+1] if i + 1 < len(all_words) else None
        
        # 1. 화자가 바뀌면 무조건 분리
        if next_w and next_w.get("speaker") != current_speaker:
            should_split = True
        
        # 2. 어미 규칙에 따른 분리 (화자가 같을 때만 추가 체크)
        if not should_split:
            ending_type = kiwi_tagger.get_ending_type(w_text)
            if ending_type == 'EF':
                should_split = True
            elif ending_type == 'EC':
                if next_w:
                    gap = float(next_w.get("start", 0)) - float(w.get("end", 0))
                    if gap >= 0.5:
                        should_split = True
                else:
                    should_split = True
        
        if should_split:
            initial_chunks.append({
                "start": float(current_words[0].get("start", 0)),
                "end": float(current_words[-1].get("end", 0)),
                "text": " ".join([(wd.get("word") or wd.get("text", "")).strip() for wd in current_words]),
                "speaker": current_speaker
            })
            current_words = []
                
    if current_words:
        initial_chunks.append({
            "start": float(current_words[0].get("start", 0)),
            "end": float(current_words[-1].get("end", 0)),
            "text": " ".join([(wd.get("word") or wd.get("text", "")).strip() for wd in current_words]),
            "speaker": current_words[-1].get("speaker")
        })

    # [2단계] 1.0초 미만의 짧은 문장은 '화자가 같을 때만' 이전 문장에 병합
    final_results = []
    for chunk in initial_chunks:
        duration = chunk["end"] - chunk["start"]
        
        if duration < 1.0 and final_results:
            # 이전 문장과 화자가 동일한 경우에만 병합
            if final_results[-1].get("speaker") == chunk.get("speaker"):
                final_results[-1]["end"] = chunk["end"]
                final_results[-1]["text"] += " " + chunk["text"]
            else:
                final_results.append(chunk)
        else:
            final_results.append(chunk)
            
    return final_results
