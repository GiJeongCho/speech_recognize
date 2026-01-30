import re
import logging
from typing import Dict, List, Any, Optional
from .kr_tag import kiwi_tagger

logger = logging.getLogger(__name__)

def refine_whisper_json(whisper_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    정제 로직 단계:
    1. 화자 단위로 그룹화 (화자가 바뀌면 분리)
    2. 각 화자 세그먼트 내에서 Kiwi의 split_into_sents 기능을 사용하여 문장 단위로 분할
    """
    raw_segments = whisper_data.get("segments") or whisper_data.get("chunks") or []
    
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

    if not all_words:
        return []

    # 1. 화자별로 연속된 단어들 그룹화
    speaker_segments = []
    current_seg = [all_words[0]]
    for i in range(1, len(all_words)):
        if all_words[i].get("speaker") == current_seg[-1].get("speaker"):
            current_seg.append(all_words[i])
        else:
            speaker_segments.append(current_seg)
            current_seg = [all_words[i]]
    speaker_segments.append(current_seg)

    final_results = []
    for seg_words in speaker_segments:
        speaker = seg_words[0].get("speaker")
        
        # 단어들을 하나의 텍스트로 합치면서 각 단어의 문자열 위치(offset) 기록
        text = ""
        word_map = [] # (start_char, end_char, word_obj)
        
        for w in seg_words:
            w_text = (w.get("word") or w.get("text", "")).strip()
            if not w_text:
                continue
            
            start_char = len(text)
            if text: # 첫 단어가 아니면 공백 추가
                text += " "
                start_char += 1
            
            text += w_text
            end_char = len(text)
            word_map.append((start_char, end_char, w))
            
        if not text:
            continue
        
        # 2. Kiwi를 사용하여 문장 분리 수행
        try:
            kiwi_sentences = kiwi_tagger.split_into_sents(text)
        except Exception as e:
            logger.error(f"Kiwi split_into_sents failed: {e}")
            kiwi_sentences = []
        
        if not kiwi_sentences:
            # 분리 실패 시 전체를 하나의 문장으로 처리
            final_results.append({
                "start": float(seg_words[0].get("start", 0)),
                "end": float(seg_words[-1].get("end", 0)),
                "text": text,
                "speaker": speaker
            })
            continue
            
        for sent in kiwi_sentences:
            # Kiwi 문장 객체의 start/end 오프셋을 기준으로 해당 문장에 포함된 단어들 찾기
            # 문장 오프셋 내에 걸쳐있는 모든 단어를 선택
            sent_words = [w for s, e, w in word_map if not (e <= sent.start or s >= sent.end)]
            
            if sent_words:
                final_results.append({
                    "start": float(sent_words[0].get("start", 0)),
                    "end": float(sent_words[-1].get("end", 0)),
                    "text": sent.text,
                    "speaker": speaker
                })
                
    return final_results
