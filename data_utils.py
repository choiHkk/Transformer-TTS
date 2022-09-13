import random
import librosa
import numpy as np
import torch
import torch.utils.data
import os

from utils import load_wav_to_torch, load_filepaths_and_text
from text import text_to_sequence
from audio_processing import melspectrogram


class TextMelLoader(torch.utils.data.Dataset):
    """
        1) loads audio,text pairs
        2) normalizes text and converts them to sequences of one-hot vectors
        3) computes mel-spectrograms from audio files.
    """
    def __init__(self, audiopaths_and_text, hparams):
        self.hparams = hparams
        self.data_path = hparams.data_path
        self.audiopaths_and_text = load_filepaths_and_text(audiopaths_and_text)
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.load_mel_from_disk = hparams.load_mel_from_disk
        self.filter_length = hparams.filter_length
        self.hop_length = hparams.hop_length
        self.win_length = hparams.win_length
        self.n_mel_channels = hparams.n_mel_channels
        self.mel_fmin = hparams.mel_fmin
        self.mel_fmax = hparams.mel_fmax
        random.seed(hparams.seed)
        random.shuffle(self.audiopaths_and_text)
    
    def get_mel_text_pair(self, audiopath_and_text):
        audiopath, text = audiopath_and_text[0], audiopath_and_text[3]
        text = self.get_text(text)
        mel = self.get_mel(os.path.join(self.data_path, audiopath))
        return (text, mel)

    def get_mel(self, filename):
        if not self.load_mel_from_disk:
            audio, sampling_rate = load_wav_to_torch(filename)
            if sampling_rate != self.sampling_rate:
                raise ValueError("{} {} SR doesn't match target {} SR".format(
                    sampling_rate, self.sampling_rate))
            audio_norm = audio / self.max_wav_value
            audio_norm, _ = librosa.effects.trim(audio_norm, top_db=35, frame_length=6000, hop_length=200)
            audio_norm = audio_norm.unsqueeze(0)
            audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
            melspec = torch.from_numpy(melspectrogram(audio_norm.squeeze(0), self.hparams))
            melspec = torch.squeeze(melspec, 0)
        else:
            melspec = torch.from_numpy(np.load(filename))
            assert melspec.size(0) == self.n_mel_channels, (
                'Mel dimension mismatch: given {}, expected {}'.format(
                    melspec.size(0), self.n_mel_channels))

        return melspec
    
    def get_text(self, text):
        text_norm = torch.IntTensor(text_to_sequence(text))
        return text_norm

    def __getitem__(self, index):
        return self.get_mel_text_pair(self.audiopaths_and_text[index])

    def __len__(self):
        return len(self.audiopaths_and_text)


class TextMelCollate():
    """ Zero-pads model inputs and targets based on number of frames per setep
    """
    def __init__(self):
        ...

    def __call__(self, batch):
        """Collate's training batch from normalized text and mel-spectrogram
        PARAMS
        ------
        batch: [text_normalized, mel_normalized]
        """
        # Right zero-pad all one-hot text sequences to max input length
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]),
            dim=0, descending=True)
        max_input_len = input_lengths[0]

        text_padded = torch.LongTensor(len(batch), max_input_len)
        text_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]][0]
            text_padded[i, :text.size(0)] = text

        # Right zero-pad mel-spec
        num_mels = batch[0][1].size(0)
        max_target_len = max([x[1].size(1) for x in batch])

        # include mel padded and gate padded
        mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
        mel_padded.zero_()
        gate_padded = torch.FloatTensor(len(batch), max_target_len)
        gate_padded.zero_()
        output_lengths = torch.LongTensor(len(batch))
        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]][1]
            mel_padded[i, :, :mel.size(1)] = mel
            gate_padded[i, mel.size(1)-1:] = 1
            output_lengths[i] = mel.size(1)

        return text_padded, input_lengths, mel_padded.transpose(1,2), gate_padded, \
            output_lengths


def segmentation_train_and_validation_text_file(hparams):
    validation_nums = hparams.validation_num
    shuffle_num = hparams.shuffle_num
    file_path = hparams.meta_files
    train_filelists_path = hparams.training_files
    validation_filelists_path = hparams.validation_files
    if not (os.path.isfile(train_filelists_path) and os.path.isfile(validation_filelists_path)):
        print("Separating train and validation text file from train.txt")
        with open(file_path, 'r', encoding='utf8') as f:
            lines = f.read().split("\n")
            lines = [line for line in lines if len(line) > 0]

            for _ in range(shuffle_num):
                random.shuffle(lines)

            trn_lines = lines[:-validation_nums]
            val_lines = lines[-validation_nums:]

            for _ in range(shuffle_num):
                random.shuffle(trn_lines)
                random.shuffle(val_lines)

            trn_infos = ""
            with open(train_filelists_path, 'w', encoding='utf8') as f_trn:
                for trn_line in trn_lines:
                    f_trn.write(trn_line+"\n")

            with open(validation_filelists_path, 'w', encoding='utf8') as f_val:
                for val_line in val_lines:
                    f_val.write(val_line+"\n")

            print(f"Done")
