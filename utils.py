from scipy.io.wavfile import read
import scipy.signal as sps
import numpy as np
import torch



def load_wav_to_torch(full_path, tsp=22050):
    sampling_rate, data = read(full_path)
    number_of_samples = round(len(data) * float(tsp) / sampling_rate)
    sampling_rate = tsp
    data = sps.resample(data, number_of_samples)
    return torch.FloatTensor(data.astype(np.float32)), sampling_rate


def load_filepaths_and_text(filename, split="|"):
    with open(filename, encoding='utf-8') as f:
        filepaths_and_text = [line.strip().split(split) for line in f]
    return filepaths_and_text


def model_parameters(model):
    param_cnt = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1_000_000
    print(f"Model Parameters: {round(param_cnt, 3)}M")