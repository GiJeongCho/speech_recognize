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
실제 오늘 한 명 면접을 보고 한 명이 더 지원을 할 면접을 볼 예정인데 그분을 뽑으면 딱 좋을 것 같아서 7년 차이시고 그때 말씀 주셨던 거 다 해보셨던 것 같아서 면접 때 확인해보고 맞는 것 같으면 일단은 여러분께 말씀드려서 최대한 빨리 보고 싶어요 당장 다음주라도 있는게 베스트인거죠 최대한 빠르게 오는게 더 좋겠죠
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
