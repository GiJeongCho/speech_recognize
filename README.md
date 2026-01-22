```
conda create -n speech_recognize python=3.9 -y
conda activate speech_recognize

# pytorch/torchaudio/cudatoolkit은 conda로 먼저
conda install pytorch=1.12.1 torchaudio=0.12.1 cudatoolkit=11.3 -c pytorch -c conda-forge

# 나머지 의존성은 uv 또는 pip로
uv pip install -r requirements.txt
uv pip install fastapi

uv add "modelscope>=1.16.0" --upgrade
uv add "huggingface_hub==0.24.7" "datasets==2.19.0"
# 또는
pip install -r requirements.txt

=======

== 그 이후 사용 시 
conda activate speech_recognize

– 일반적 사용 시 

ssh -L 8010:localhost:8010 elice-vm
ppsnipa5763@@

cd /home/pps-nipa/NIQ/fish/speech_recognize
conda activate speech_recognize
uv run uvicorn src.api:app --host 0.0.0.0 --port 8011 --reload

http://localhost:8011/docs
http://localhost:8011/redoc

== uv lock만들기
rm uv.lock
uv lock


==포드맨 사용 (docker 대체) ==
podman build -t pps/speech_recognize:v0.0.1 -f Dockerfile . 

podman build prune -a 								| 빌드과정 생성된 캐쉬 날리기 


podman build –no-cache -t pps/speech_recognize:v0.0.1 -f Dockerfile .	| 이전 기록 무시하고 새로 맘

podman run -d -p 8011:8011 --restart always --name whisper pps/whisper-stt:v0.0.1 		| 시작

podman logs -f pps/speech_recognize:v0.0.1
podman rm -f pps/speech_recognize:v0.0.1
podman rmi -f pps/speech_recognize:v0.0.1
```