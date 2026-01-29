import re
from kiwipiepy import Kiwi
import logging

# 로거 설정 [[memory:6804125]]
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    # Kiwi 초기화 ('sbg' 대신 권장되는 'cong' 모델 사용)
    kiwi = Kiwi(
        num_workers=0, 
        model_path=None, 
        load_default_dict=True, 
        integrate_allomorph=True, 
        model_type='cong', 
        typos=None, 
        typo_cost_threshold=2.5
    )

    # 테스트 데이터
    texts = [
"""
아니라 개발자들이 같이 진행을 하는 저는 시스템 아키텍처 써주신 거 보고 에이전트 아키텍처 지금 학원에 있는데 랭그레이크에 대해서 공부를 추가적으로 해야겠다는 생각을 하고 금일 중으로 마무리되지 않을까 싶어요
"""
    ]

    print("=== Kiwi 형태소 분석 테스트 ===\n")

    for text in texts:
        print(f"분석 문장: {text}")
        
        # 1. 형태소 분석 (tokenize)
        tokens = kiwi.tokenize(text)
        result = " | ".join([f"{token.form}/{token.tag}" for token in tokens])
        print(f"분석 결과: {result}")
        
        # 2. 문장 분리 (split_into_sents)
        if len(text) > 15:
            sents = kiwi.split_into_sents(text)
            print(f"문장 분리: {[s.text for s in sents]}")
        
        print("-" * 30)

except Exception as e:
    # 예외 발생 시 로깅 [[memory:6804125]]
    logger.exception(f"Kiwi 실행 중 에러가 발생했습니다: {e}")
