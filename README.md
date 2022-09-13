## Introduction
1. Tacotron2 오픈 소스와 pytorch 기본 내장 transformer를 활용하여 Transformer-TTS를 간단 구현하고 한국어 데이터셋(KSS)을 사용해 빠르게 학습합니다.
2. 본 레포지토리에서 Tacotron2에서 제안하는 BCE loss를 그대로 사용하면 stop token이 제대로 예측되지 않기 때문에 Transformer-TTS 논문에서 제안하는 coefficient를 사용하여 Stop Token Layer를 학습합니다..
3. conda 환경으로 진행해도 무방하지만 본 레포지토리에서는 docker 환경만 제공합니다. 기본적으로 ubuntu에 docker, nvidia-docker가 설치되었다고 가정합니다.
4. GPU, CUDA 종류에 따라 Dockerfile 상단 torch image 수정이 필요할 수도 있습니다.
5. 별도의 pre-processing 과정은 필요하지 않습니다.
6. Batch Size 32 기준, 약 3만 step 정도에서 모델이 말을 할 수 있을 정도가 됩니다.

## Dataset
1. download dataset - https://www.kaggle.com/datasets/bryanpark/korean-single-speaker-speech-dataset
2. `unzip /path/to/the/kss.zip -d /path/to/the/kss`
3. `mkdir /path/to/the/Transformer-tts/data/dataset`
4. `mv /path/to/the/kss.zip /path/to/the/Transformer-tts/data/dataset`

## Docker build
1. `cd /path/to/the/Transformer-tts`
2. `docker build --tag Transformer-tts:latest .`

## Training
1. `nvidia-docker run -it --name 'Transformer-tts' -v /path/to/Transformer-tts:/home/work/Transformer-tts --ipc=host --privileged Transformer-tts:latest`
2. `pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113`
3. `cd /home/work/Transformer-tts`
4. `ln -s /home/work/Transformer-tts/data/dataset/kss`
5. `python train.py -g 0 -o data -l kss_v1 -d ./kss -c`
6. `python train.py -g 0 -o data -l kss_v1 -d ./kss -c ./data/kss_v1/model_state/checkpoint_<step>`
7. arguments
  * -g : gpu number
  * -o : output directory
  * -i : log directory
  * -d : data directory
  * -c : checkpoint path with step number
8. (OPTIONAL) `tensorboard --logdir=outdir/logdir`

## Tensorboard losses


## Tensorboard Stats


## Reference
1. [Neural Speech Synthesis with Transformer Network](https://arxiv.org/abs/1809.08895)
2. [Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions](https://arxiv.org/abs/1712.05884)
