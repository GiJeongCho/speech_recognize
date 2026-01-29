import logging
from kiwipiepy import Kiwi
from typing import List

logger = logging.getLogger(__name__)

class KiwiTagger:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(KiwiTagger, cls).__new__(cls)
            try:
                # Kiwi 초기화 (싱글톤)
                cls._instance.kiwi = Kiwi(
                    num_workers=0, 
                    integrate_allomorph=True, 
                    model_type='cong', 
                    typo_cost_threshold=2.5
                )
                logger.info("Kiwi initialized successfully within KiwiTagger class.")
            except Exception as e:
                # [[memory:6804125]] 예외 발생 시 로깅
                logger.error(f"Failed to initialize Kiwi in KiwiTagger: {e}")
                cls._instance.kiwi = None
        return cls._instance

    def is_terminal_ending(self, text: str) -> bool:
        """
        텍스트에 종결 어미(EF)가 포함되어 있는지 확인합니다.
        """
        if not text.strip() or self.kiwi is None:
            return False
            
        try:
            analysis = self.kiwi.analyze(text)
            if analysis:
                tokens = analysis[0][0]
                # 태그 중 EF(종결 어미)가 하나라도 있으면 문장 종료로 간주
                return any(t.tag == 'EF' for t in tokens)
        except Exception as e:
            # [[memory:6804125]] 예외 발생 시 로깅
            logger.error(f"Kiwi analysis failed for text '{text}': {e}")
            # 분석 실패 시 마침표 등으로 보조 판단
            return any(p in text for p in [".", "!", "?"])
        
        return False

# 싱글톤 인스턴스 생성
kiwi_tagger = KiwiTagger()
