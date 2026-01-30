import logging
from typing import Dict, List, Any, Final

from .kr_tag import kiwi_tagger

logger = logging.getLogger(__name__)

# Constants
MIN_SPEAKER_DURATION: Final[float] = 0.5
SPEAKER_VERY_SHORT: Final[str] = "very_short"
SPEAKER_UNKNOWN: Final[str] = "unknown"


def refine_whisper_json(whisper_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Whisper 출력을 Kiwi를 사용하여 문장 단위로 정제하고 짧은 구간을 분류합니다.

    로직 단계:
    1. 모든 단어(words)를 시간 순서대로 추출합니다.
    2. 전체 단어를 하나의 텍스트로 합치며 각 단어의 문자열 오프셋을 기록합니다.
    3. Kiwi의 `split_into_sents` 기능을 사용하여 문장 단위로 분할합니다.
    4. 분할된 문장의 길이가 0.5초 미만인 경우 화자명을 'very_short'로 설정합니다.

    Args:
        whisper_data: Whisper 엔진에서 반환된 원본 JSON 데이터.

    Returns:
        List[Dict[str, Any]]: 정제된 문장 단위 조각 리스트.
    """
    raw_segments = whisper_data.get("segments") or whisper_data.get("chunks") or []
    
    all_words: List[Dict[str, Any]] = []
    for seg in raw_segments:
        words = seg.get("words", [])
        if not words:
            # 단어 단위 정보가 없는 경우 세그먼트 자체를 하나의 단어처럼 처리
            all_words.append({
                "start": float(seg.get("start", 0)),
                "end": float(seg.get("end", 0)),
                "word": seg.get("text", "").strip(),
            })
        else:
            all_words.extend(words)

    if not all_words:
        return []

    # 단어들을 하나의 텍스트로 합치면서 각 단어의 문자열 위치(offset) 기록
    full_text: str = ""
    word_map: List[tuple[int, int, Dict[str, Any]]] = []
    
    for w in all_words:
        w_text = (w.get("word") or w.get("text", "")).strip()
        if not w_text:
            continue
        
        start_char = len(full_text)
        if full_text:  # 첫 단어가 아니면 공백 추가
            full_text += " "
            start_char += 1
        
        full_text += w_text
        end_char = len(full_text)
        word_map.append((start_char, end_char, w))
            
    if not full_text:
        return []
    
    # 2. Kiwi를 사용하여 문장 분리 수행
    try:
        kiwi_sentences = kiwi_tagger.split_into_sents(full_text)
    except Exception as e:
        # [[memory:6804125]] 예외 발생 시 로깅
        logger.error(f"Kiwi split_into_sents failed during refine: {e}")
        kiwi_sentences = []
    
    final_results: List[Dict[str, Any]] = []
    
    if not kiwi_sentences:
        # 분리 실패 시 전체를 하나의 세그먼트로 처리
        start_time = float(all_words[0].get("start", 0))
        end_time = float(all_words[-1].get("end", 0))
        duration = end_time - start_time
        
        final_results.append({
            "start": start_time,
            "end": end_time,
            "text": full_text,
            "speaker": SPEAKER_VERY_SHORT if duration < MIN_SPEAKER_DURATION else SPEAKER_UNKNOWN
        })
    else:
        for sent in kiwi_sentences:
            # 문장 오프셋 내에 걸쳐있는 단어들 매핑
            sent_words = [w for s, e, w in word_map if not (e <= sent.start or s >= sent.end)]
            
            if sent_words:
                start_time = float(sent_words[0].get("start", 0))
                end_time = float(sent_words[-1].get("end", 0))
                duration = end_time - start_time
                
                final_results.append({
                    "start": start_time,
                    "end": end_time,
                    "text": sent.text,
                    "speaker": SPEAKER_VERY_SHORT if duration < MIN_SPEAKER_DURATION else SPEAKER_UNKNOWN
                })
                
    return final_results

