import re
from typing import Dict, List, Any
from .kr_tag import kiwi_tagger

def refine_whisper_json(whisper_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Kiwi 문장 분리를 사용하여 문장을 나누되, 
    데이터를 누락하거나 무리하게 합치지 않고 1문장 단위로 정제합니다.
    """
    raw_segments = whisper_data.get("segments") or whisper_data.get("chunks") or []
    
    initial_chunks = []
    for seg in raw_segments:
        text = seg.get("text", "").strip()
        if not text: continue
        
        words = seg.get("words", [])
        sents = kiwi_tagger.split_into_sents(text)
        
        if not words or not sents:
            initial_chunks.append({
                "start": float(seg.get("start", 0)),
                "end": float(seg.get("end", 0)),
                "text": text
            })
            continue
            
        # 단어들을 각 문장에 배분
        word_idx = 0
        for sent in sents:
            sent_text = sent.text.strip()
            sent_words = []
            sent_clean = re.sub(r'\s+', '', sent_text)
            accum_text = ""
            
            while word_idx < len(words):
                w = words[word_idx]
                w_text = (w.get("word") or w.get("text", "")).strip()
                sent_words.append(w)
                accum_text += re.sub(r'\s+', '', w_text)
                word_idx += 1
                if len(accum_text) >= len(sent_clean):
                    break
            
            if sent_words:
                initial_chunks.append({
                    "start": float(sent_words[0].get("start", 0)),
                    "end": float(sent_words[-1].get("end", 0)),
                    "text": sent_text
                })

    # [단계 2] 아주 짧은 문장 처리 (침묵 간격이 0.5초 이내일 때만 이전 문장과 병합)
    final_results = []
    if initial_chunks:
        current = initial_chunks[0]
        for next_chunk in initial_chunks[1:]:
            duration = current["end"] - current["start"]
            gap = next_chunk["start"] - current["end"]
            
            # 현재 문장이 너무 짧고(0.8초 미만), 다음 문장과의 간격이 좁을 때만 병합
            if duration < 0.8 and gap < 0.5:
                current["end"] = next_chunk["end"]
                current["text"] += " " + next_chunk["text"]
            else:
                final_results.append(current)
                current = next_chunk
        final_results.append(current)
    
    return final_results
