import logging
import re
from kiwipiepy import Kiwi
from typing import List, Any

logger = logging.getLogger(__name__)

class KiwiTagger:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(KiwiTagger, cls).__new__(cls)
            try:
                # 사용자 설정에 따라 'cong' 모델 사용
                cls._instance.kiwi = Kiwi(
                    num_workers=0, 
                    integrate_allomorph=True, 
                    model_type='cong', 
                    typo_cost_threshold=2.5
                )
                logger.info("Kiwi initialized successfully with model_type='cong'.")
            except Exception as e:
                # [[memory:6804125]] 예외 발생 시 로깅
                logger.error(f"Failed to initialize Kiwi in KiwiTagger: {e}")
                cls._instance.kiwi = None
        return cls._instance

    def split_into_sents(self, text: str) -> List[Any]:
        """
        Kiwi의 문장 분리 기능을 사용하여 텍스트를 문장 단위로 나눕니다.
        """
        if not text.strip() or self.kiwi is None:
            return []
        try:
            return self.kiwi.split_into_sents(text)
        except Exception as e:
            # [[memory:6804125]]
            logger.error(f"Kiwi split_into_sents failed: {e}")
            return []

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
                return any(t.tag == 'EF' for t in tokens)
        except Exception as e:
            # [[memory:6804125]]
            logger.error(f"Kiwi analysis failed for text '{text}': {e}")
            return any(p in text for p in [".", "!", "?"])
        return False

# 싱글톤 인스턴스 생성
kiwi_tagger = KiwiTagger()
