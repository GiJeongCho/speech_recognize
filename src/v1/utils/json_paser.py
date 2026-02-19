import logging
from dataclasses import dataclass
from typing import Dict, List, Any, Final, Optional, Tuple, TypedDict

from .kr_tag import kiwi_tagger

logger = logging.getLogger(__name__)

# Constants
MIN_SPEAKER_DURATION: Final[float] = 0.5
SPEAKER_VERY_SHORT: Final[str] = "very_short"
SPEAKER_UNKNOWN: Final[str] = "unknown"

class RefinedChunk(TypedDict):
    start: float
    end: float
    text: str
    speaker: str

@dataclass
class WordItem:
    start: float
    end: float
    text: str
    speaker: str
    explicit_speaker: bool


@dataclass
class SpeakerGroup:
    speaker: str
    words: List[WordItem]
    has_explicit_speaker: bool


@dataclass
class ChunkInternal:
    start: float
    end: float
    text: str
    speaker: str
    has_explicit_speaker: bool


def _normalize_speaker(raw_speaker: Optional[object]) -> Optional[str]:
    if raw_speaker is None:
        return None
    speaker = str(raw_speaker).strip()
    return speaker or None


def extract_segments(whisper_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw_segments: Optional[List[Dict[str, Any]]] = whisper_data.get("segments") or whisper_data.get("chunks")

    if raw_segments is None and "result" in whisper_data:
        result_payload = whisper_data["result"]
        if isinstance(result_payload, dict):
            raw_segments = result_payload.get("segments") or result_payload.get("chunks")

    return raw_segments or []


def _extract_words(raw_segments: List[Dict[str, Any]]) -> List[WordItem]:
    all_words: List[WordItem] = []

    for seg in raw_segments:
        if not isinstance(seg, dict):
            continue

        seg_speaker_raw = _normalize_speaker(seg.get("speaker"))
        seg_speaker = seg_speaker_raw or SPEAKER_UNKNOWN
        seg_has_speaker = seg_speaker_raw is not None

        words = seg.get("words", [])
        if not words:
            seg_text = str(seg.get("text", "")).strip()
            if not seg_text:
                continue
            all_words.append(
                WordItem(
                    start=float(seg.get("start", 0)),
                    end=float(seg.get("end", 0)),
                    text=seg_text,
                    speaker=seg_speaker,
                    explicit_speaker=seg_has_speaker,
                )
            )
            continue

        for w in words:
            if not isinstance(w, dict):
                continue
            w_text = str(w.get("word") or w.get("text", "")).strip()
            if not w_text:
                continue

            word_speaker_raw = _normalize_speaker(w.get("speaker"))
            if word_speaker_raw is not None:
                speaker = word_speaker_raw
                explicit = True
            elif seg_has_speaker:
                speaker = seg_speaker
                explicit = True
            else:
                speaker = SPEAKER_UNKNOWN
                explicit = False

            all_words.append(
                WordItem(
                    start=float(w.get("start", 0)),
                    end=float(w.get("end", 0)),
                    text=w_text,
                    speaker=speaker,
                    explicit_speaker=explicit,
                )
            )

    return all_words


def _group_words_by_speaker(words: List[WordItem]) -> List[SpeakerGroup]:
    groups: List[SpeakerGroup] = []

    for word in words:
        if not groups or groups[-1].speaker != word.speaker:
            groups.append(
                SpeakerGroup(
                    speaker=word.speaker,
                    words=[word],
                    has_explicit_speaker=word.explicit_speaker,
                )
            )
            continue

        groups[-1].words.append(word)
        if word.explicit_speaker:
            groups[-1].has_explicit_speaker = True

    return groups


def _build_text_and_map(words: List[WordItem]) -> Tuple[str, List[Tuple[int, int, WordItem]]]:
    full_text = ""
    word_map: List[Tuple[int, int, WordItem]] = []

    for w in words:
        if not w.text:
            continue

        start_char = len(full_text)
        if full_text:
            full_text += " "
            start_char += 1

        full_text += w.text
        end_char = len(full_text)
        word_map.append((start_char, end_char, w))

    return full_text, word_map


def _resolve_speaker(duration: float, group: SpeakerGroup) -> str:
    if group.has_explicit_speaker:
        return group.speaker
    return SPEAKER_VERY_SHORT if duration < MIN_SPEAKER_DURATION else SPEAKER_UNKNOWN


def _split_group_by_kiwi(group: SpeakerGroup) -> List[ChunkInternal]:
    full_text, word_map = _build_text_and_map(group.words)
    if not full_text:
        return []

    try:
        kiwi_sentences = kiwi_tagger.split_into_sents(full_text)
    except Exception as e:
        # [[memory:6804125]] 예외 발생 시 로깅
        logger.error(f"Kiwi split_into_sents failed during refine: {e}")
        kiwi_sentences = []

    chunks: List[ChunkInternal] = []
    if not kiwi_sentences:
        start_time = float(group.words[0].start)
        end_time = float(group.words[-1].end)
        duration = end_time - start_time
        chunks.append(
            ChunkInternal(
                start=start_time,
                end=end_time,
                text=full_text,
                speaker=_resolve_speaker(duration, group),
                has_explicit_speaker=group.has_explicit_speaker,
            )
        )
        return chunks

    for sent in kiwi_sentences:
        sent_words = [w for s, e, w in word_map if not (e <= sent.start or s >= sent.end)]
        if not sent_words:
            continue

        start_time = float(sent_words[0].start)
        end_time = float(sent_words[-1].end)
        duration = end_time - start_time

        chunks.append(
            ChunkInternal(
                start=start_time,
                end=end_time,
                text=sent.text,
                speaker=_resolve_speaker(duration, group),
                has_explicit_speaker=group.has_explicit_speaker,
            )
        )

    return chunks


def _merge_short_segments(chunks: List[ChunkInternal]) -> List[ChunkInternal]:
    if not chunks:
        return []

    final_results: List[ChunkInternal] = []
    # 리스트 복사 (뒤쪽 병합 시 다음 청크를 수정해야 하므로)
    processing_chunks = chunks[:]
    
    i = 0
    n = len(processing_chunks)
    
    while i < n:
        curr = processing_chunks[i]
        duration = curr.end - curr.start
        
        # 0.2초 초과면 정상 청크로 간주하여 결과에 추가
        if duration > 0.2:
            final_results.append(curr)
            i += 1
            continue
            
        # --- 0.2초 이하인 경우: 앞뒤 거리 비교하여 더 가까운 쪽에 병합 ---
        
        # 1. 앞쪽 청크와 거리 계산
        prev = final_results[-1] if final_results else None
        gap_prev = float('inf')
        if prev:
            gap_prev = curr.start - prev.end
            
        # 2. 뒤쪽 청크와 거리 계산
        next_chunk = None
        gap_next = float('inf')
        if i + 1 < n:
            next_chunk = processing_chunks[i+1]
            gap_next = next_chunk.start - curr.end
            
        # 3. 앞뒤 모두 없으면(단독 0.2초 이하) 그냥 추가
        if not prev and not next_chunk:
            final_results.append(curr)
            i += 1
            continue

        # 4. 거리 비교 및 병합 (같으면 앞쪽 우선)
        if gap_prev <= gap_next:
            # 앞쪽에 병합 (앞 화자에게 할당)
            prev.end = curr.end
            prev.text = f"{prev.text} {curr.text}".strip()
        else:
            # 뒤쪽에 병합 (뒤 화자에게 할당)
            # 다음 청크의 시작 시간을 당기고 텍스트를 앞에 붙임
            next_chunk.start = curr.start
            next_chunk.text = f"{curr.text} {next_chunk.text}".strip()
            
        # 현재 청크(curr)는 병합되어 사라졌으므로 final_results에 추가하지 않고 넘어감
        i += 1

    return final_results


def refine_whisper_json(whisper_data: Dict[str, Any]) -> List[RefinedChunk]:
    """Whisper 출력을 Kiwi를 사용하여 문장 단위로 정제하고 화자 변경을 반영합니다.

    (최상위 segments/chunks 키 또는 result.segments 구조 지원)

    로직 단계:
    1. 모든 단어(words)를 시간 순서대로 추출합니다.
    2. 전체 단어를 하나의 텍스트로 합치며 각 단어의 문자열 오프셋을 기록합니다.
    3. Kiwi의 `split_into_sents` 기능을 사용하여 문장 단위로 분할합니다.
    4. 화자가 바뀌는 지점을 기준으로 중간 분할합니다.
    5. 화자 정보가 없을 때만 0.5초 미만 문장을 'very_short'로 분류합니다.

    Args:
        whisper_data (Dict[str, Any]): Whisper 엔진에서 반환된 원본 JSON 데이터.

    Returns:
        List[RefinedChunk]: 정제된 문장 단위 조각 리스트.
    """
    raw_segments = extract_segments(whisper_data)
    all_words = _extract_words(raw_segments)
    if not all_words:
        return []

    speaker_groups = _group_words_by_speaker(all_words)
    split_chunks: List[ChunkInternal] = []
    for group in speaker_groups:
        split_chunks.extend(_split_group_by_kiwi(group))

    merged_chunks = _merge_short_segments(split_chunks)
    return [
        {
            "start": chunk.start,
            "end": chunk.end,
            "text": chunk.text,
            "speaker": chunk.speaker,
        }
        for chunk in merged_chunks
    ]
