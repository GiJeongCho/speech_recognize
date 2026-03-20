# uv run modelscope download --model iic/speech_eres2netv2_sv_zh-cn_16k-common --cache_dir src/resoursces/models

import os
from modelscope.hub.snapshot_download import snapshot_download

def download_eres2net_v2():
    # 현재 파일 위치: src/resoursces/test/Eres2NetV2_download.py
    # 목표 모델 디렉토리: src/resoursces/models
    # 상위(test) -> 상위(resoursces) -> models
    current_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.abspath(os.path.join(current_dir, "..", "models"))
    
    # 모델 ID (ERes2NetV2)
    model_id = 'iic/speech_eres2netv2_sv_zh-cn_16k-common'
    
    print(f"Start downloading {model_id}...")
    print(f"Target directory: {models_dir}")
    
    # 디렉토리 생성
    os.makedirs(models_dir, exist_ok=True)
    
    try:
        # 모델 다운로드
        # cache_dir를 지정하면 cache_dir/iic/speech_eres2netv2_sv_zh-cn_16k 형태로 저장됩니다.
        model_path = snapshot_download(model_id, cache_dir=models_dir)
        print(f"Download completed successfully!")
        print(f"Model stored at: {model_path}")
    except Exception as e:
        print(f"Failed to download model: {e}")

if __name__ == "__main__":
    download_eres2net_v2()
