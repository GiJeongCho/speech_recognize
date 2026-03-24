import os
from modelscope.hub.snapshot_download import snapshot_download

def download_eres2net():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.abspath(os.path.join(current_dir, "..", "models"))
    os.makedirs(models_dir, exist_ok=True)

    model_id = 'iic/speech_eres2net_base_sv_zh-cn_3dspeaker_16k'
    print(f"Downloading {model_id}...")
    print(f"Target directory: {models_dir}")
    try:
        model_path = snapshot_download(model_id, cache_dir=models_dir)
        print(f"Done: {model_path}")
    except Exception as e:
        print(f"Failed: {e}")

if __name__ == "__main__":
    download_eres2net()
